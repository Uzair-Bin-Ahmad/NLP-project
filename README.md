# DuoRAG Assistant

DuoRAG Assistant is a personal Retrieval-Augmented Generation (RAG) chatbot built for an NLP course project. It answers questions about **Uzair Bin Ahmad** and **Muhammad Zain** using a custom personal dataset derived from their academic and professional profiles.

## Main Features

- Custom personal dataset for two Software Engineering students
- Document loading from TXT and optional PDF files
- Text preprocessing and chunking
- Sentence Transformer embeddings
- FAISS vector database
- Semantic retrieval of relevant chunks
- Gemini-based context-aware response generation
- Prompt engineering to reduce hallucination
- Conversation history using Streamlit session state
- Source file display for retrieved context
- Streamlit web interface
- Ready for Streamlit Community Cloud deployment

## Architecture

```text
Personal Documents
      ↓
Document Loader
      ↓
Preprocessing + Chunking
      ↓
Sentence Transformer Embeddings
      ↓
FAISS Vector Database
      ↓
User Question
      ↓
Semantic Retrieval
      ↓
Retrieved Context + Chat History + Prompt
      ↓
Gemini LLM
      ↓
Context-Aware Answer
```

## Project Structure

```text
DuoRAGBot/
├── app.py
├── rag.py
├── ingest.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   ├── uzair_profile.txt
│   └── zain_profile.txt
└── vectorstore/        # generated automatically
```

## Technologies

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Google Gemini API
- PyPDF
- NumPy

## Dataset

The dataset contains custom profile information about two students, including:

- Education
- Programming and frontend skills
- Backend and database skills
- Machine Learning and NLP coursework
- Projects
- Professional experience where available
- Learning strengths and technical focus

Direct personal contact information has intentionally been excluded from the public chatbot dataset.

## Local Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Gemini API key

Copy `.env.example` to `.env` and add your key:

```text
GEMINI_API_KEY=your_real_api_key
```

### 4. Build the vector database

```bash
python ingest.py
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Select `app.py` as the main file.
5. Open the app's **Settings → Secrets**.
6. Add:

```toml
GEMINI_API_KEY="your_real_api_key"
```

7. Save and redeploy.
8. Test the public URL before submission.

The app automatically builds the FAISS index on first launch if the `vectorstore` folder is not already present.

## Example Questions

- What projects has Uzair built?
- What technical skills does Zain have?
- Compare Uzair and Zain's frontend skills.
- What NLP or ML topics have they studied?
- Which student has professional frontend experience?
- Tell me about Zain's crypto trading bot.
- What React projects has Uzair completed?
- What is Uzair's favorite football club?  
  Expected behavior: the chatbot should state that this information is not in the knowledge base.

## How the RAG Pipeline Works

1. **Document Loading**: TXT/PDF documents are loaded from the `data` folder.
2. **Preprocessing**: Extra whitespace is normalized.
3. **Chunking**: Long documents are divided into overlapping chunks.
4. **Embedding Generation**: Each chunk is converted into a numerical vector using `all-MiniLM-L6-v2`.
5. **Vector Storage**: The vectors are stored in FAISS.
6. **Retrieval**: A user question is embedded and compared with stored vectors to retrieve the most relevant chunks.
7. **Prompt Engineering**: Retrieved context, recent chat history, and the question are placed in a prompt that instructs the model not to invent personal facts.
8. **LLM Generation**: Gemini generates the final answer from the retrieved context.
9. **History Maintenance**: Streamlit session state stores previous user and assistant messages during the session.

## Viva Explanation

A concise explanation:

> Our system is a personal RAG chatbot for two Software Engineering students. We created a custom dataset from their academic and professional profiles. The documents are preprocessed and divided into overlapping chunks. A Sentence Transformer converts those chunks into embeddings, which are stored in a FAISS vector database. When a user asks a question, the same embedding model converts the question into a vector and FAISS retrieves the most semantically relevant chunks. These chunks are inserted into an engineered prompt and sent to Gemini. The prompt instructs Gemini to answer only from retrieved context and to avoid inventing unsupported personal information. Streamlit provides the user interface and maintains chat history using session state.

## Important Security Note

Do not commit `.env` or your real Gemini API key to GitHub. Use Streamlit Secrets for deployment.
