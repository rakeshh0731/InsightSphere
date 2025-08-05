# InsightSphere - AI-Powered Analytics Engine

This project is a full-stack, Retrieval-Augmented Generation (RAG) chatbot application. It features a FastAPI backend that uses LangChain for orchestration and a React-based frontend for user interaction. It is designed to answer questions about a specific knowledge base, which is provided as a set of documents (currently supporting PDFs and TXT files) in a `knowledge_base` directory.

The application leverages a local, open-source sentence-transformer model for creating text embeddings and the high-speed Groq API for Large Language Model (LLM) inference, providing fast and contextually-aware responses.

## Tech Stack

### Backend
- **Framework**: FastAPI
- **Orchestration**: LangChain
- **LLM Inference**: Groq (Llama 3)
- **Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store**: ChromaDB
- **Document Loading**: PyPDFLoader, TextLoader

### Frontend
- **Framework**: React (with Vite)
- **Styling**: CSS

## How It Works

The application follows a standard RAG pipeline to provide answers based on the content of documents you provide:

1.  **Document Loading**: On startup, the backend API loads all PDF and TXT documents from the `InsightSphere_server/knowledge_base/` directory.
2.  **Text Splitting**: The documents are split into smaller, manageable text chunks.
3.  **Embedding Generation**: A local sentence-transformer model (`all-MiniLM-L6-v2`) is used to convert each text chunk into a numerical vector (embedding). This model is automatically downloaded on the first run.
4.  **Vector Storage**: These embeddings are stored in a local ChromaDB vector store, which is persisted in the `InsightSphere_server/chroma_db/` directory. This allows for efficient similarity searches.
5.  **Retrieval**: When a user asks a question, the application first retrieves the most relevant text chunks from the vector store based on semantic similarity to the question.
6.  **Generation**: The retrieved context and the user's question are passed to the Groq API (using the Llama 3 model), which generates a final, human-like answer.
7.  **Conversation History**: The chatbot maintains a history for each session, allowing for follow-up questions that reference previous parts of the conversation.

## Project Structure

```
InsightSphere/
├── .gitignore
├── InsightSphere_server/      # Backend FastAPI application
│   ├── api_layer.py
│   ├── client.py
│   ├── requirements.txt
│   ├── .env
│   └── knowledge_base/      # Place your PDF and TXT files here
├── InsightSphere_UI/          # Frontend React application
│   ├── public/
│   └── src/
└── README.md                  # This file
```

## Setup and Installation

### Prerequisites
- Python 3.9+
- Node.js and npm

### 1. Clone the Repository
```sh
git clone <your-github-repository-url>
cd InsightSphere
```

### 2. Backend Setup (InsightSphere_server)

Navigate to the server directory:
```sh
cd InsightSphere_server
```

Create and activate a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

Install Python dependencies:
```sh
pip install -r requirements.txt
```

Set up environment variables. Create a file named `.env` in the `InsightSphere_server` directory and add your Groq API key:
```
GROQ_API_KEY="gsk_..."
```

Add your knowledge base. Create a directory named `knowledge_base` inside `InsightSphere_server` and place your PDF and/or TXT documents inside it.

### 3. Frontend Setup (InsightSphere_UI)

Navigate to the UI directory from the project root:
```sh
cd InsightSphere_UI
```

Install Node.js dependencies:
```sh
npm install
```

## Running the Application

You need to run both the backend and frontend servers in separate terminals.

### 1. Start the Backend API

From the `InsightSphere_server` directory:
```sh
python api_layer.py
```
**First-Time Run:**
- The first time you run the application, it will automatically download the `all-MiniLM-L6-v2` embedding model from Hugging Face. This is a one-time process and may take a few minutes depending on your internet connection. The model will be saved in a local `InsightSphere_server/all-MiniLM-L6-v2` folder.
- The application will also create the ChromaDB vector store from your documents and persist it in the `InsightSphere_server/chroma_db` directory.

The API will be available at `http://localhost:8000`.

### 2. Start the Frontend UI

From the `InsightSphere_UI` directory:
```sh
npm run dev
```
The React application will start, and you can access it in your browser at the URL provided (usually `http://localhost:5173`).

## API Endpoints

You can also interact with the API directly using a tool like `curl` or any API client.

1.  **Start a New Chat Session**
    ```sh
    curl -X POST http://localhost:8000/session/start
    ```
    *Response:* `{"chatSessionId": "some-unique-id"}`

2.  **Send a Message**
    Use the `chatSessionId` from the previous step.
    ```sh
    curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
        "chatSessionId": "some-unique-id",
        "message": "What is the main purpose of this document?"
    }'
    ```

3.  **Close the Session**
    ```sh
    curl -X POST http://localhost:8000/session/close \
    -H "Content-Type: application/json" \
    -d '{
        "chatSessionId": "some-unique-id"
    }'
    ```