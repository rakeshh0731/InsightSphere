import asyncio
import ssl
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# We import our new LangChain-based classes
from client import Configuration, LangChainClient, ChatSession

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. Initializes shared components on startup
    and cleans them up on shutdown.
    """
    # --- Startup Logic ---
    logging.info("Application startup: Initializing LangChain components...")
    try:
        # 1. Load configuration
        config = Configuration()

        # 2. Initialize the main LangChain client
        # This class holds the LLM and embedding models.
        langchain_client = LangChainClient(config)

        # 3. Setup the vector store and retriever (heavy I/O operation)
        # This will load the PDF, create embeddings, and persist them if not already done.
        retriever = langchain_client.setup_vector_store("knowledge_base")

        # 4. Store shared, initialized components in app state
        app.state.langchain_client = langchain_client
        app.state.retriever = retriever
        app.state.active_sessions = {}  # Dictionary to hold active chat sessions

        logging.info("LangChain components initialized and ready.")

    except Exception as e:
        logging.error(f"Fatal error during startup: {e}", exc_info=True)
        # Prevent app from starting if initialization fails
        raise

    yield  # The application runs while the lifespan context is active

    # --- Shutdown Logic ---
    logging.info("Application shutdown: Cleaning up resources...")
    active_sessions: dict | None = getattr(app.state, "active_sessions", None)
    if active_sessions:
        logging.info(f"Clearing {len(active_sessions)} active session(s)...")
        # Clean up history from the class-level store
        for session in active_sessions.values():
            session.close()
        active_sessions.clear()
        logging.info("Active sessions cleared.")

    logging.info("Cleanup complete.")


app = FastAPI(
    title="InsightSphere API",
    description="An API to interact with the InsightSphere AI-Powered Analytics Engine.",
    version="2.0.0",
    lifespan=lifespan,
)


# Pydantic models (remain the same for frontend compatibility)
class StartSessionResponse(BaseModel):
    chatSessionId: str = Field(..., alias="chatSessionId")

class SessionRequest(BaseModel):
    chatSessionId: str = Field(..., alias="chatSessionId")

class CloseSessionResponse(BaseModel):
    message: str

class ClearSessionResponse(BaseModel):
    message: str
    chatSessionId: str = Field(..., alias="chatSessionId")

class ChatRequest(BaseModel):
    chatSessionId: str = Field(..., alias="chatSessionId")
    message: str

class ChatResponse(BaseModel):
    chatSessionId: str = Field(..., alias="chatSessionId")
    reply: str


@app.post("/session/start", response_model=StartSessionResponse, status_code=201)
async def start_session():
    """Starts a new chat session and returns a unique session ID."""
    client: LangChainClient | None = getattr(app.state, "langchain_client", None)
    retriever: any = getattr(app.state, "retriever", None)

    if not client or not retriever:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not available due to a startup initialization error."
        )

    # Create a new ChatSession instance, passing the shared LLM and retriever
    chat_session = ChatSession(llm=client.llm, retriever=retriever)
    session_id = chat_session.session_id
    app.state.active_sessions[session_id] = chat_session

    logging.info(f"Started new chat session with ID: {session_id}")
    return StartSessionResponse(chatSessionId=session_id)


@app.post("/session/clear", response_model=ClearSessionResponse)
async def clear_session(request: SessionRequest):
    """Clears the conversation history for a session, effectively resetting it."""
    active_sessions: dict[str, ChatSession] = getattr(app.state, "active_sessions", {})
    session_id = request.chatSessionId
    chat_session = active_sessions.get(session_id)

    if not chat_session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    chat_session.reset()

    logging.info(f"Cleared/Reset session with ID: {session_id}")
    return ClearSessionResponse(
        message=f"Session {session_id} history cleared successfully.",
        chatSessionId=session_id
    )


@app.post("/session/close", response_model=CloseSessionResponse)
async def close_session(request: SessionRequest):
    """Closes a chat session and removes it entirely from memory."""
    active_sessions: dict[str, ChatSession] = getattr(app.state, "active_sessions", {})
    session_id = request.chatSessionId

    if session_id in active_sessions:
        session_to_close = active_sessions.pop(session_id)
        session_to_close.close()  # Clean up its history from the store
        logging.info(f"Closed and cleared session with ID: {session_id}")
        return CloseSessionResponse(message=f"Session {session_id} closed successfully.")
    else:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Receives a user message and returns the assistant's reply."""
    active_sessions: dict[str, ChatSession] = getattr(app.state, "active_sessions", {})
    chat_session = active_sessions.get(request.chatSessionId)

    if not chat_session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.chatSessionId}' not found. Please start a new session."
        )

    try:
        logging.info(f"Received message for session '{request.chatSessionId}': '{request.message}'")
        reply_text = await chat_session.chat(request.message)
        logging.info(f"Sending reply for session '{request.chatSessionId}': '{reply_text}'")
        return ChatResponse(chatSessionId=request.chatSessionId, reply=reply_text)
    except Exception as e:
        logging.error(f"Error during chat processing for session '{request.chatSessionId}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the message."
        )


def main():
    """Synchronous entry point to run the API server."""
    uvicorn.run("api_layer:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()