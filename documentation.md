```markdown
# T"AI" - Technical Documentation

This document provides a comprehensive technical overview of the T"AI" (Your Personalized AI Teaching Assistant) system. It is intended for software engineers and developers who require a detailed understanding of the codebase, architecture, and workflows.

---

## Overview

T"AI" integrates Large Language Models (LLMs), vector stores, and other AI services into a Streamlit-based application that enhances the teaching and learning experience. It provides:

- **Role-based User Management**: Professors can create/manage courses; students can access and interact with them.
- **Document Ingestion & Vectorization**: PDFs and text files are uploaded, parsed, embedded, and stored in a FAISS vector store.
- **AI-Powered Tools**:
  - **Question Answering (Retrieval-Augmented)**: Students ask questions; the LLM responds with context from ingested docs.
  - **Summarization, MCQs, Flashcards**: Automatically generate summaries, quizzes, and study aids from course materials.
  - **Podcast Generation**: Convert text-based materials into a podcast script, then to audio via gTTS.
- **YouTube Integration**: Fetch transcripts from YouTube videos, integrate into vector store, and recommend relevant videos.
- **Analytics & Insights**: Log and classify student questions, produce visualizations and reports from CSV data.

---

## Architecture

**Front-End**:  
- **Streamlit**: Renders UI components, handles file uploads, user interactions, and session states.

**Back-End**:  
- **Database & ORM**: SQLite via SQLAlchemy models (`User`, `Course`, `CourseFile`, `StudentQuestion`).
- **LLM Integration**: OpenAI GPT-based models through LangChain (with retrieval chains, prompt templates, and embeddings).
- **Vector Store**: FAISS for semantic search and retrieval.
- **External Tools**:
  - **PyPDF2** for PDF extraction.
  - **AssemblyAI** for transcription (optional).
  - **gTTS** for text-to-speech.
  - **Plotly**, **Matplotlib**, and **WordCloud** for analytics.
  
**Data Flow**:
1. **Ingestion**: PDFs → Extract text → Chunking → Embeddings → Vector Store.
2. **Q&A**: Question → Retrieval from Vector Store → LLM → Answer.
3. **Analytics**: Questions logged in CSV → Data Visualization & Topic Classification.

---

## Deployment & Running the Application

T"AI" can be run locally or inside a Docker container.

### Environment Setup

1. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file in the project root with:
   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   ASSEMBLYAI_API_KEY=your-assemblyai-api-key
   ```
   The `OPENAI_API_KEY` is mandatory. `ASSEMBLYAI_API_KEY` is needed for YouTube transcription functionality.

### Running Locally

```bash
streamlit run application.py
```

Access the UI at `http://localhost:8501`.

### Running with Docker

A `Dockerfile` is provided for containerized deployment. You can build and run it as follows:

1. **Build the Image**:
   ```bash
   docker build -t tai-assistant .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 8501:8501 --env-file .env tai-assistant
   ```

Access the UI at `http://localhost:8501`.

---

## File Structure

```plaintext
project/
├─ application.py         # Main application (UI, logic, authentication)
├─ application1.py        # Additional styling, static pages (About, Contact, Home)
├─ base.py                # SQLAlchemy base
├─ models.py              # ORM models (User, Course, etc.)
├─ database.py            # DB session creation
├─ data/                  # CSV files for analytics
├─ img/                   # Image assets
├─ Dockerfile             # Docker build instructions
├─ requirements.txt       # Dependencies
└─ ... (other assets)
```

---

## Database Models

- **User**: Stores `username`, `password_hash`, `role` (`professor` or `student`).
- **Course**: Linked to a `User` (professor), stores `youtube_link` optionally.
- **CourseFile**: Binary data (PDFs, transcripts) associated with a `Course`.
- **StudentQuestion**: Logs student queries (`question`, `timestamp`).

All relationships are defined in `models.py` using SQLAlchemy ORM.

---

## Core Functionalities

### Document Processing & Vectorization

- Upload PDFs/text files.
- Extract text using `PyPDF2`.
- Chunk text with LangChain’s `CharacterTextSplitter`.
- Embed using `OpenAIEmbeddings`.
- Index chunks in a FAISS vector store.

### Q&A with Documents

- Retrieves the top `k` relevant documents from the FAISS store.
- Uses LLM via LangChain’s `RetrievalQA` chain.
- Answers are grounded in course materials.

### Summaries, MCQs, and Flashcards

- **Summarization**: Uses `load_summarize_chain` to create a map-reduce summary.
- **MCQs and Flashcards**: Prompt templates in LangChain generate exam-like questions or flashcards from course content.

### Podcast Generation

- Extract and combine selected text from course materials.
- Prompt GPT to create a narrative podcast script.
- Convert script to MP3 via `gTTS`.

### YouTube Integration

- Enter a YouTube link in the professor dashboard.
- Download and transcribe audio (if API keys provided).
- Store transcript as a `CourseFile` and integrate into the vector store.

### Analytics & Insights

- **CSV Logging**: Student questions appended to `data/ml_grouped_topics_questions.csv`.
- **Visualization**: Plotly for bar/pie charts, WordCloud for keywords.
- **Report Generation**: GPT creates a textual report summarizing insights from the CSV.

---

## Technical Details

- **LLM Model**: GPT-4 or GPT-3.5-turbo (adjustable in code).
- **Chunk Size & Overlap**: Default chunking is 2,000 chars with 100 chars overlap. Tunable based on use-case.
- **Caching**: In-memory caching via `InMemoryCache`. Consider persistent caching for scaling.
- **Error Handling & Logging**: Python `logging` with `INFO` and `ERROR` levels. Try/except blocks around I/O and LLM calls.

---

## Security & Access Control

- Passwords hashed using `bcrypt`.
- Basic role checks (professors vs students).
- Environment variables and `.env` for secret management.

---

## Extensibility

- **Alternative Vector Stores**: Pinecone, Weaviate, or ChromaDB.
- **Additional Models**: Replace `OpenAIEmbeddings` or `ChatOpenAI` with other LLMs (e.g., Anthropic).
- **Additional File Formats**: DOCX, HTML, and other loaders.
- **Scaling**: Containerized deployments, use cloud-based managed databases, integrate authentication providers.

---

## Conclusion

T"AI" provides a robust, AI-driven educational platform with easy extensibility. With a clean architecture, role-based access, and a spectrum of features (Q&A, summaries, MCQs, podcasts, analytics), it sets a foundation for advanced EdTech experiences. The `.env` file and optional Docker setup allow flexible deployment in various environments.
```
