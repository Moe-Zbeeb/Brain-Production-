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

## A one scroll for some selected main code/fucntions read with f strings and comments ( not actual code here !!)  
```markdown
# Full Code Documentation

This document provides comprehensive, line-by-line and function-level documentation for the provided codebase. The code is separated into main files (`application.py`, `application1.py`, `models.py`, `base.py`, and `database.py`) and utilizes various frameworks and libraries such as Streamlit, SQLAlchemy, OpenAI, and LangChain.

**Note:** The following documentation assumes the code structure and content as previously presented. All code snippets are annotated with comments, docstrings, and explanations for clarity. References to environment variables and keys assume that `.env` files are correctly set as mentioned in previous documentation.

---

## `base.py`

**Purpose**: Defines the base SQLAlchemy declarative class used by models.

```python
# base.py

from sqlalchemy.orm import declarative_base

# The Base class for all SQLAlchemy models. 
# All ORM models will inherit from this Base.
Base = declarative_base()
```

**Explanation**:  
- `declarative_base()` returns a base class that our model classes will inherit from. This is standard SQLAlchemy practice.

---

## `models.py`

**Purpose**: Defines the SQLAlchemy ORM models for `User`, `Course`, `CourseFile`, and `StudentQuestion`. Relationships between models are also defined here.

```python
# models.py

from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from base import Base
from datetime import datetime
import bcrypt

class User(Base):
    """
    User model representing application users.
    Columns:
        id (int): Primary key user ID.
        username (str): Unique username for login.
        password_hash (str): Bcrypt-hashed password.
        role (str): User role, either 'professor' or 'student'.
    Relationships:
        courses (list[Course]): Courses created by this user if professor.
        questions (list[StudentQuestion]): Questions asked by this user if student.
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)

    courses = relationship("Course", back_populates="professor", cascade="all, delete-orphan")
    questions = relationship("StudentQuestion", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        """
        Hash and store the user password using bcrypt.
        Args:
            password (str): Plain-text password.
        """
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password: str) -> bool:
        """
        Verify the provided password against the stored bcrypt hash.
        Args:
            password (str): Plain-text password input.
        Returns:
            bool: True if password matches, else False.
        """
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))


class Course(Base):
    """
    Course model representing a university course managed by a professor.
    Columns:
        id (int): Primary key course ID.
        name (str): Unique course name.
        professor_id (int): Foreign key to the User (professor).
        youtube_link (str): Optional YouTube link associated with the course.
    Relationships:
        professor (User): The professor who created/owns this course.
        files (list[CourseFile]): Files (PDFs/transcripts) associated with this course.
        questions (list[StudentQuestion]): Questions asked by students about this course.
    """
    __tablename__ = 'courses'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    professor_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    youtube_link = Column(String, nullable=True)

    professor = relationship("User", back_populates="courses")
    files = relationship("CourseFile", back_populates="course", cascade="all, delete-orphan")
    questions = relationship("StudentQuestion", back_populates="course", cascade="all, delete-orphan")


class CourseFile(Base):
    """
    CourseFile model representing stored files associated with a course.
    Columns:
        id (int): Primary key file ID.
        filename (str): Name of the file.
        data (bytes): Binary content of the file (PDF text, transcripts, etc.).
        course_id (int): Foreign key to the associated Course.
    Relationships:
        course (Course): The course this file is associated with.
    """
    __tablename__ = 'course_files'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    data = Column(LargeBinary, nullable=False)
    course_id = Column(Integer, ForeignKey('courses.id'), nullable=False)
    course = relationship("Course", back_populates="files")


class StudentQuestion(Base):
    """
    StudentQuestion model logs questions asked by students.
    Columns:
        id (int): Primary key question ID.
        user_id (int): Foreign key to User (the student who asked the question).
        course_id (int): Foreign key to Course.
        question (str): The actual question text.
        timestamp (DateTime): Time when the question was asked.
    Relationships:
        user (User): The student who asked the question.
        course (Course): The course the question refers to.
    """
    __tablename__ = 'student_questions'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    course_id = Column(Integer, ForeignKey('courses.id'), nullable=False)
    question = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="questions")
    course = relationship("Course", back_populates="questions")
```

---

## `database.py`

**Purpose**: Creates a SQLAlchemy session to interact with the database.

```python
# database.py

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from base import Base
import os

# By default, using SQLite in memory or a file-based SQLite.
# Adjust the connection string as needed.
DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they do not exist yet.
Base.metadata.create_all(bind=engine)
```

**Explanation**:  
- `SessionLocal` is a session factory. 
- `Base.metadata.create_all()` ensures tables are created on startup if missing.

---

## `application1.py`

**Purpose**: Provides UI styling functions, page layouts (About, Contact, Home pages), and helper utilities.

```python
# application1.py

import streamlit as st
import base64
import os
import io
from PIL import Image

def set_overlay_bg_image(image_path: str) -> str:
    """
    Convert an image file to a base64-encoded data URI for use in CSS backgrounds.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: base64 data URI for the image.
    """
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        encoded_img = base64.b64encode(data).decode()
        return f"data:image/png;base64,{encoded_img}"
    else:
        st.error(f"Image not found at: {image_path}")
        return ""

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode a PIL image to a base64 string.
    Args:
        image (Image.Image): PIL Image object.
    Returns:
        str: base64-encoded string of the image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def inject_css():
    """
    Injects custom CSS for header, footer, layout styling. Also inserts a footer snippet.
    """
    footer = """
    <style>
        footer {
            background-color: #FFFFFF;
            color: #333333;
            padding: 10px 20px;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            font-size: 25px;
            margin-top: 50px;
            border-top: 1px solid #CCCCCC;
        }
        .footer-left {
            text-align: center;
            line-height: 1.5;
        }
    </style>

    <footer>
        <div class="footer-left">
            <span>The ultimate tool for understanding the information that matters most to you, built with GPTs</span>
        </div>
    </footer>
    """
    st.markdown(footer, unsafe_allow_html=True)

    # Additional header/logo area styling
    # Using inline CSS for demonstration

def inject_css2():
    """
    Inject CSS specifically for the About/Contact/Home section styling, 
    along with footer design and layout.
    """
    st.markdown("""
        <style>
        /* Additional styling for footer, about sections, etc. */
        .footer {
            background-color: #1C1C44;
            color: white;
            padding: 50px;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

def about_page():
    """
    Renders the About page with a banner, textual content, and styling.
    Uses helper functions for background images and styling.
    """
    inject_css()
    # ... About page UI rendering, omitted here for brevity but same pattern as shown.

def contact_page():
    """
    Renders the Contact page with location, phone, and email info.
    """
    inject_css()
    # ... Contact page UI layout and styling.

def encode_video_to_base64(video_path: str) -> str:
    """
    Encode a local video file to a base64 string for embedding in HTML.
    Args:
        video_path (str): Path to the video file.
    Returns:
        str: Base64 encoded video data.
    """
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")
```

---

## `application.py`

**Purpose**: Main application file that orchestrates user authentication, course creation, document ingestion, LLM interactions, analytics, and UI layouts for both professor and student users. It is quite large, so we will document each major section and function.

```python
# application.py

import streamlit as st
import PyPDF2
import logging
import tempfile
import os
import base64
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import csv
import urllib.request
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from database import SessionLocal
from gtts import gTTS
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import subprocess
import assemblyai as aai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # Ensure correct package is installed (User code)
from langchain.schema import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.cache import InMemoryCache
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from models import User, Course, CourseFile, StudentQuestion
from application1 import about_page, contact_page, inject_css, inject_css2, set_overlay_bg_image, encode_image_to_base64

load_dotenv()
aai.settings.api_key = "YOUR_ASSEMBLYAI_KEY"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Verify OpenAI API key presence
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Ensure it's set in the environment variables.")

logging.basicConfig(level=logging.INFO)

# Initialize in-memory cache for LangChain
cache = InMemoryCache()

# Initialize database session
session_db = SessionLocal()

# Initialize LLM
try:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )
    logging.info("Successfully connected to OpenAI LLM.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI LLM: {str(e)}")
    st.error(f"Failed to initialize OpenAI LLM: {str(e)}")
    st.stop()


class LangchainHandler:
    """
    A handler class encapsulating document loading, vector store creation, 
    and interactions with LLM for retrieval, summarization, and generation tasks.
    """
    def __init__(self, llm):
        self.llm = llm

    def load_document(self, file_path: str):
        """
        Load a document (PDF or text) into LangChain Documents.
        Splits into chunks for better processing.

        Args:
            file_path (str): Path to the PDF or text file.
        Returns:
            List[Document]: A list of Document objects with text chunks.
        """
        try:
            logging.info(f"Loading document from: {file_path}")
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            raw_docs = loader.load()
            logging.info(f"Loaded {len(raw_docs)} documents.")

            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            docs = text_splitter.split_documents(raw_docs)
            logging.info(f"Split the document into {len(docs)} chunks.")
            return docs
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            return []

    def create_vector_store(self, docs):
        """
        Create a FAISS vector store from provided documents.

        Args:
            docs (List[Document]): Document objects to be indexed.
        Returns:
            FAISS: A FAISS vector store instance.
        """
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.from_documents(docs, embeddings)
            logging.info("Created vector store from documents.")
            return vector_store
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            return None

    def get_response(self, vector_store, question: str):
        """
        Retrieve documents from vector store and use LLM to answer a question academically.

        Args:
            vector_store (FAISS): Vector store for retrieval.
            question (str): User question.

        Returns:
            str: Academic response string based on retrieved context.
        """
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            prompt_template = """
            You are an academic assistant. Provide a clear, concise academic answer based on these documents.
            Question: {question}
            Documents:
            {context}

            Response:
            """
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["question", "context"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            response = qa_chain.run(question)
            logging.info("Generated academic response to user question.")
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Sorry, I couldn't process your question at the moment."

    def summarize_documents(self, documents):
        """
        Summarize a set of documents using LLM's map-reduce summarization chain.

        Args:
            documents (List[Document]): Document objects to summarize.
        Returns:
            str: A textual summary of the documents.
        """
        try:
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            summary = chain.run(documents)
            logging.info("Generated summary of course materials.")
            return summary.strip()
        except Exception as e:
            logging.error(f"Error during summarization: {str(e)}")
            return "Sorry, I couldn't summarize the course materials."

    def generate_mcq_questions(self, documents, num_questions=10):
        """
        Generate multiple-choice questions from documents.

        Args:
            documents (List[Document]): Documents as source.
            num_questions (int): Number of MCQs to generate.

        Returns:
            str: MCQs text.
        """
        try:
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            template = """
            You are a teacher. Based on the following text, generate {num_questions} multiple-choice questions.
            Text:
            {text}
            """
            prompt = PromptTemplate(
                input_variables=["num_questions", "text"],
                template=template
            )

            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            response = llm_chain.run(num_questions=num_questions, text=combined_text)
            logging.info("Generated MCQs from course materials.")
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating MCQs: {str(e)}")
            return "Sorry, couldn't generate MCQs."

    def generate_flashcards(self, documents, num_flashcards=20):
        """
        Generate flashcards from documents.

        Args:
            documents (List[Document]): Source documents.
            num_flashcards (int): Number of flashcards.

        Returns:
            str: Q&A formatted flashcards.
        """
        try:
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            template = """
            You are a teacher. Based on this text, generate {num_flashcards} flashcards.
            Each flashcard:
            Q: ...
            A: ...
            """
            prompt = PromptTemplate(
                input_variables=["num_flashcards", "text"],
                template=template
            )

            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            response = llm_chain.run(num_flashcards=num_flashcards, text=combined_text)
            logging.info("Generated flashcards.")
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating flashcards: {str(e)}")
            return "Sorry, couldn't generate flashcards."

    def generate_podcast_script(self, extracted_text: str, openai_api_key: str) -> str:
        """
        Generate a podcast script based on provided extracted text.

        Args:
            extracted_text (str): Source text to create a podcast script from.
            openai_api_key (str): OpenAI key for the LLM.

        Returns:
            str: Podcast script.
        """
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert podcaster. Create a detailed engaging script from this content."),
                ("user", "{content}")
            ])

            langchain_llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                openai_api_key=openai_api_key
            )
            chain = LLMChain(llm=langchain_llm, prompt=prompt_template)
            script = chain.run(content=extracted_text)
            logging.info("Generated podcast script.")
            return script.strip()
        except Exception as e:
            logging.error(f"Error generating podcast script: {e}")
            return ""

    def generate_podcast_audio(self, script: str, output_filename: str) -> str:
        """
        Convert text script to speech using gTTS and save as MP3.

        Args:
            script (str): Podcast script text.
            output_filename (str): Output MP3 filename.

        Returns:
            str: File path to the saved audio file.
        """
        try:
            tts = gTTS(text=script, lang='en')
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, output_filename)
            tts.save(audio_path)
            logging.info(f'Audio content saved to "{audio_path}"')
            return audio_path
        except Exception as e:
            logging.error(f"Error converting text to speech: {e}")
            return ""

# Initialize LangchainHandler
langchain_handler = LangchainHandler(llm=llm)

def validate_youtube_url(url: str) -> bool:
    """
    Validate if a URL is a YouTube video link.
    Args:
        url (str): URL string.
    Returns:
        bool: True if valid YouTube URL else False.
    """
    pattern = r"^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$"
    return re.match(pattern, url) is not None

def download_audio_yt_dlp(video_url: str, output_dir: str) -> bool:
    """
    Download audio from YouTube using yt-dlp.
    Args:
        video_url (str): YouTube video URL.
        output_dir (str): Output directory for audio file.
    Returns:
        bool: True if successful, False if error.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "%(title)s.%(ext)s")
        command = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--user-agent", "Mozilla/5.0",
            "--output", output_file,
            video_url
        ]
        subprocess.run(command, check=True)
        return True
    except Exception as e:
        logging.error(f"Error downloading audio: {e}")
        return False

def process_youtube_links(youtube_links, course, output_dir="transcripts"):
    """
    Download and transcribe YouTube links, store transcripts.
    Args:
        youtube_links (list[str]): List of YouTube URLs.
        course (Course): Course object to associate transcripts with.
        output_dir (str): Directory to save transcripts.
    Returns:
        dict: Mapping audio_file to transcript text.
    """
    os.makedirs(output_dir, exist_ok=True)
    transcriber = aai.Transcriber()
    transcripts = {}

    for link in youtube_links:
        if not validate_youtube_url(link):
            logging.info(f"Invalid URL: {link}")
            continue
        if not download_audio_yt_dlp(link, output_dir):
            continue
        try:
            # Find downloaded MP3
            audio_file = next((f for f in os.listdir(output_dir) if f.endswith(".mp3")), None)
            if audio_file:
                audio_path = os.path.join(output_dir, audio_file)
                config = aai.TranscriptionConfig(speaker_labels=True)
                transcript = transcriber.transcribe(audio_path, config)
                
                transcript_file = os.path.join(output_dir, f"{audio_file}_transcript.txt")
                with open(transcript_file, "w") as f:
                    for utterance in transcript.utterances:
                        f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")
                
                transcripts[audio_file] = "\n".join(
                    [f"Speaker {utt.speaker}: {utt.text}" for utt in transcript.utterances]
                )

                # Add to DB
                transcript_filename = f"{os.path.splitext(audio_file)[0]}_transcript.txt"
                course_file = CourseFile(
                    filename=transcript_filename,
                    data=transcripts[audio_file].encode('utf-8'),
                    course_id=course.id
                )
                session_db.add(course_file)
                session_db.commit()
                logging.info(f"Transcript added to course {course.name}.")
        except Exception as e:
            logging.error(f"Error processing {link}: {e}")
    return transcripts

def set_bg_image():
    """
    Inject CSS to set a white background for the app.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background-color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

def signup_page():
    """
    Render Sign Up page. Allows new users (professors or students) to create accounts.
    """
    inject_css()
    # Render UI and handle form submission for signup
    # Validates username uniqueness, password match, and saves user to DB.

def login_page():
    """
    Render Login page. Authenticates existing users.
    """
    inject_css()
    # Renders login form and checks credentials against DB.
    # On success, store user in session and navigate to dashboard.

def professor_page():
    """
    Professor Dashboard:
    - Create/Manage Courses
    - Upload Files
    - Add YouTube links
    - View Insights and Delete Courses
    """
    # Side navigation and UI components.
    # Calls helper functions to handle forms and data loading.
    pass

def home_page():
    """
    Home Page:
    Introduction, features overview, promotional video, etc.
    """
    inject_css()
    # Render the marketing/home content, features, and demos.

def classify_topic(question: str) -> str:
    """
    Classify a student's question into a machine learning topic based on keyword matching.
    Args:
        question (str)
    Returns:
        str: Identified topic or 'General' if no match.
    """
    keyword_topic_map = {
        'regression': 'Regression',
        'classification': 'Classification',
        # Add more keywords...
    }
    question_lower = question.lower()
    for keyword, topic in keyword_topic_map.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', question_lower):
            return topic
    return 'General'

def update_course_csv(csv_file_path: str, question: str, topic: str):
    """
    Append a new question and topic to CSV for analytics.
    Args:
        csv_file_path (str): Path to CSV.
        question (str): Student question.
        topic (str): Classified topic.
    """
    try:
        directory = os.path.dirname(csv_file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Topic', 'Question']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({'Topic': topic, 'Question': question})
    except Exception as e:
        logging.error(f"Failed to update CSV: {e}")

def generate_youtube_keyword(api_key: str, query: str) -> str:
    """
    Use LLM to refine a query into a YouTube search keyword.
    Args:
        api_key (str)
        query (str)
    Returns:
        str: Refined YouTube search keyword.
    """
    chat = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
    messages = [
        SystemMessage(content="You are an expert at generating YouTube search keywords."),
        HumanMessage(content=f"Suggest a good YouTube search keyword for this topic: {query}")
    ]
    try:
        response = chat.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"

def search_youtube(keyword: str, num_results=10) -> list:
    """
    Search YouTube for videos matching the keyword.
    Args:
        keyword (str)
        num_results (int)
    Returns:
        list[str]: Video URLs
    """
    search_keyword = keyword.replace(" ", "+")
    url = f"https://www.youtube.com/results?search_query={search_keyword}"
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    video_ids = re.findall(r"watch\?v=(\S{11})", str(soup))
    if video_ids:
        return [f"https://www.youtube.com/watch?v={v}" for v in video_ids[:num_results]]
    return []

def download_transcripts(video_links: list, folder_path="transcripts") -> dict:
    """
    Simulate transcript downloads for given video links.
    (In reality, integrate with a transcription service)
    Args:
        video_links (list[str])
        folder_path (str)
    Returns:
        dict: video_id to transcript
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    transcripts = {}
    # Placeholder logic as explained in code comment
    for link in video_links:
        video_id = link.split("=")[-1]
        transcript_path = os.path.join(folder_path, f"transcript_{video_id}.txt")
        fake_transcript = f"Transcript for video {video_id}"
        with open(transcript_path, 'w', encoding='utf-8') as file:
            file.write(fake_transcript)
        transcripts[video_id] = fake_transcript
    return transcripts

def embed_transcripts(transcripts: dict) -> dict:
    """
    Embed transcripts using SentenceTransformer model.
    Args:
        transcripts (dict): {video_id: transcript_text}
    Returns:
        dict: {video_id: embedding_vector}
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = {}
    for video_id, transcript in transcripts.items():
        embedding = model.encode(transcript, convert_to_numpy=True)
        embeddings[video_id] = embedding
    return embeddings

def recommend_video(query_embedding: np.ndarray, video_embeddings: dict) -> tuple:
    """
    Recommend the most relevant video based on query embedding similarity.
    Args:
        query_embedding (np.ndarray)
        video_embeddings (dict): video_id to embedding.
    Returns:
        (video_id, similarity_score)
    """
    video_ids = list(video_embeddings.keys())
    all_embeddings = np.array(list(video_embeddings.values()))
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    top_index = similarities.argmax()
    return video_ids[top_index], similarities[top_index]

def student_page():
    """
    Student Dashboard:
    - View courses
    - Generate flashcards, MCQs, summarization
    - Chat with documents (Q&A)
    - Podcast generation from PDFs
    - YouTube video recommendations
    """
    # Renders UI, calls respective functions. 
    # Each form submission triggers LLM-based generations or retrieval.

# Additional helper functions for generating charts, wordclouds, and CSV analysis.
def generate_pie_chart(df: pd.DataFrame):
    # Create a pie chart of topics.
    pass

def generate_bar_chart(df: pd.DataFrame):
    # Create a bar chart of topic distribution.
    pass

def generate_wordcloud(df: pd.DataFrame):
    # Generate a word cloud from questions.
    pass

def generate_csv_report(csv_file_path: str) -> str:
    # Use LLM to generate a textual report of CSV data.
    pass

def main():
    """
    Main entrypoint for Streamlit app.
    Handles page routing, user login state, and navigation.
    """
    if "user" not in st.session_state:
        st.session_state.user = None
    if "page" not in st.session_state:
        st.session_state.page = "home"

    page_mapping = {
        "home": home_page,
        "signup": signup_page,
        "login": login_page,
        "contact": contact_page,
        "about": about_page,
        "professor": professor_page,
        "student": student_page,
    }

    # Navigation buttons at the top
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        if col1.button("Home"):
            st.session_state.page = "home"
        if col2.button("Sign Up"):
            st.session_state.page = "signup"
        if col3.button("Login"):
            st.session_state.page = "login"
        if col4.button("Contact"):
            st.session_state.page = "contact"
        if col5.button("About"):
            st.session_state.page = "about"

    # If user is logged in and selected "dashboard" page:
    if st.session_state.page == "dashboard":
        if st.session_state.user and hasattr(st.session_state.user, "role"):
            if st.session_state.user.role == "professor":
                professor_page()
            else:
                student_page()
        else:
            st.error("User not logged in or role missing.")
    else:
        # Render the selected page from the mapping
        page_mapping.get(st.session_state.page, home_page)()

if __name__ == "__main__":
    main()
```

---

## Summary

- **`base.py`**: Defines the SQLAlchemy base model.
- **`models.py`**: Contains the database ORM models (`User`, `Course`, `CourseFile`, `StudentQuestion`).
- **`database.py`**: Handles database initialization and session creation.
- **`application1.py`**: Provides auxiliary pages (Home, About, Contact), CSS injection, and image/video base64 utilities.
- **`application.py`**: The main application file, containing:
  - User Authentication (Login, Signup)
  - Role-based navigation (Professor Dashboard, Student Dashboard)
  - Course management and file uploads
  - Document ingestion, vector indexing, and LLM-based Q&A
  - Content generation: Summaries, MCQs, Flashcards, Podcast scripts
  - YouTube integration and transcription handling
  - Analytics (CSV logging, topic classification, plotting, and word clouds)

This comprehensive annotation provides insights into each function, class, and code block, enabling developers to easily navigate, maintain, and extend the codebase.
```
