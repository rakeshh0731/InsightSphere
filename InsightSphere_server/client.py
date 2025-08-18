import logging
import os
import httpx
import uuid
from typing import Any, List, Dict

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EmptyRetriever(BaseRetriever):
    """A retriever that always returns no documents."""

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        """Sync method, returns an empty list."""
        return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager
    ) -> List[Document]:
        """Async method, returns an empty list."""
        return []


class Configuration:
    """Manages configuration and environment variables."""

    def __init__(self) -> None:
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")


class LangChainClient:
    """
    Manages the one-time setup for the LangChain RAG pipeline.
    This includes loading documents, creating embeddings, and setting up the vector store.
    """

    def __init__(self, config: Configuration):
        self.config = config
        self.vector_store_path = "./chroma_db"
        # Create a custom httpx client that disables SSL verification.
        # This is a quick fix for corporate firewalls like Zscaler.
        # For production, you should use `verify="/path/to/zscaler.pem"`
        async_client = httpx.AsyncClient(verify=False)

        self.llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=self.config.groq_api_key,
            http_async_client=async_client,
        )

        # Define model path and name for local embeddings
        local_model_path = "./all-MiniLM-L6-v2"
        hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Check if the model exists locally. If not, download it.
        # This makes the setup self-contained and runs only once.
        if not os.path.exists(local_model_path):
            logging.info(
                f"Local model not found. Downloading '{hf_model_name}' to '{local_model_path}'..."
            )
            logging.info(
                "This will happen only once. Please ensure you have an internet connection."
            )
            try:
                from sentence_transformers import SentenceTransformer

                downloader = SentenceTransformer(hf_model_name)
                downloader.save(local_model_path)
                logging.info("Model downloaded successfully.")
            except Exception as e:
                logging.error(
                    f"Fatal: Failed to download embedding model. Please check internet or firewall. Error: {e}",
                    exc_info=True,
                )
                raise

        # Load the embedding model from the (now guaranteed) local path.
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=local_model_path, model_kwargs={"device": "cpu"}
        )

    def setup_vector_store(self, docs_path: str) -> Any:
        """
        Sets up or updates the ChromaDB vector store.
        - On first run, creates the store from documents in `docs_path`.
        - On subsequent runs, loads the existing store and adds any new documents
          found in `docs_path`.
        - If no documents are found, it returns a dummy retriever.
        """
        # Step 1: Load all current documents from the source directory.
        all_documents = []
        if os.path.exists(docs_path):
            logging.info(f"Scanning for documents in '{docs_path}'...")
            # Load PDF files
            pdf_loader = DirectoryLoader(
                docs_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True,
            )
            pdf_docs = pdf_loader.load()
            logging.info(f"Found {len(pdf_docs)} PDF documents.")

            # Load TXT files
            txt_loader = DirectoryLoader(
                docs_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True,
            )
            txt_docs = txt_loader.load()
            logging.info(f"Found {len(txt_docs)} text documents.")
            all_documents = pdf_docs + txt_docs
        else:
            logging.warning(f"Knowledge base directory '{docs_path}' not found. No documents will be loaded.")

        vector_store_exists = os.path.exists(self.vector_store_path)

        # Step 2: Handle case where vector store exists. Load it and add new docs.
        if vector_store_exists:
            logging.info(f"Loading existing vector store from {self.vector_store_path}")
            vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embedding_function,
            )

            # Check for new documents to add
            if all_documents:
                existing_docs_metadata = vector_store.get(include=["metadatas"])
                existing_sources = {
                    m["source"] for m in existing_docs_metadata["metadatas"] if "source" in m
                }
                new_documents = [
                    doc for doc in all_documents if doc.metadata["source"] not in existing_sources
                ]

                if new_documents:
                    logging.info(f"Found {len(new_documents)} new document(s) to add to the vector store.")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs_to_add = text_splitter.split_documents(new_documents)
                    vector_store.add_documents(docs_to_add)
                    logging.info("Successfully added new documents to ChromaDB and persisted changes.")
                else:
                    logging.info("Vector store is already up-to-date.")
            return vector_store.as_retriever()

        # Step 3: Handle case where vector store does NOT exist.
        else:
            if not all_documents:
                logging.warning(
                    "No documents found in knowledge base. The application will start, "
                    "but the chatbot will operate without document context. "
                    "To add context, place PDF or TXT files in the 'knowledge_base' directory and restart."
                )
                return EmptyRetriever()

            logging.info(f"Creating new vector store from {len(all_documents)} document(s).")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(all_documents)
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_function,
                persist_directory=self.vector_store_path,
            )
            logging.info("Vector store created and persisted.")
            return vector_store.as_retriever()


class ChatSession:
    """
    Orchestrates the interaction for a single conversation, managing its own history.
    """
    # In-memory store for chat histories, keyed by session ID.
    _store: Dict[str, BaseChatMessageHistory] = {}

    def __init__(self, llm: ChatGroq, retriever: Any):
        self.session_id = str(uuid.uuid4())
        self.llm = llm
        self.retriever = retriever
        self.conversational_rag_chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Creates the full RAG chain with history awareness."""
        # 1. Contextualize Question Prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # 2. Answering Prompt
        qa_system_prompt = (
            "You are an expert assistant for InsightSphere. Use the following pieces of "
            "retrieved context to answer the question. If you don't know the answer, "
            "just say that you don't know. Keep your answers concise and helpful. Always answer based on the provided context."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # 3. Full RAG Chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 4. Add History Management
        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieves or creates a chat history for a given session ID."""
        if session_id not in self._store:
            from langchain_community.chat_message_histories import (
                ChatMessageHistory,
            )
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    async def chat(self, user_input: str) -> str:
        """Handles a single user message and returns the assistant's response."""
        logging.info(f"Invoking chain for session {self.session_id}...")
        response = await self.conversational_rag_chain.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": self.session_id}},
        )
        return response.get("answer", "I'm sorry, I encountered an issue.")

    def reset(self) -> None:
        """Clears the conversation history for this session."""
        if self.session_id in self._store:
            self._store[self.session_id].clear()
            logging.info(f"History for session {self.session_id} has been cleared.")

    def close(self) -> None:
        """Removes the session history from the store."""
        if self.session_id in self._store:
            del self._store[self.session_id]
            logging.info(f"History for session {self.session_id} has been removed.")
