

## Overview

This repository contains a collection of Python modules, Streamlit applications, and database models aimed at creating an interactive educational platform enhanced by AI. It brings together several functionalities, including:

1. **User Authentication & Role Management:** Professors and students can sign up and log in.
2. **Course Management:** Professors can create and manage courses, upload course materials (PDFs, text files), and add associated YouTube links.
3. **AI-Powered Content Generation:**
   - **PDF-to-Podcast Conversion:** Extract text from PDFs, generate a podcast script using GPT-4, and convert it to audio.
   - **Q&A with Course Documents:** Use embeddings and vector databases to answer student queries academically based on uploaded course materials.
   - **Flashcards and MCQs Generation:** Automatically produce flashcards and multiple-choice questions from course materials.
   - **Summaries:** Summarize course documents to help students grasp key points quickly.
4. **YouTube Integration:** Assist with generating refined YouTube search keywords using OpenAI and recommend the most relevant video based on transcript embeddings.
5. **Data Insights:** Collect and categorize student questions by topic, visualize them (charts, word clouds), and generate a comprehensive LLM-based report to help instructors improve the course.

This platform leverages the following technologies:

- **Streamlit** for web interface and user interaction.
- **SQLAlchemy** and SQLite for data persistence.
- **OpenAI’s GPT-4** model (via LangChain) for generating content and understanding documents.
- **PyPDF2** for PDF text extraction.
- **gTTS** for text-to-speech conversion of podcast scripts.
- **FAISS** and **SentenceTransformers** for vector embeddings, semantic search, and document retrieval.
- **AssemblyAI** (optional) for YouTube transcript generation.
- **yt-dlp** for audio extraction from YouTube videos.

## Repository Structure

A brief explanation of key files:

- **`podcast.py`**:   
  Handles PDF extraction, podcast script generation, and audio conversion. It uses GPT-4 (via LangChain) to transform PDF contents into a podcast script and then `gTTS` to convert text to speech.

- **`database.py`**, **`base.py`**, **`models.py`**:   
  Define database models (Users, Courses, CourseFiles, StudentQuestions) and handle the database connection and setup. Built with SQLAlchemy and SQLite.

- **`application.py`**:  
  The main Streamlit application that ties together all functionalities: user login, signup, course management, question asking, MCQs/flashcards generation, summarization, embedding-based Q&A, and YouTube integration. This is where the main UI flow is defined.

- **`application1.py`**:  
  Contains additional UI components, styling, and page templates (e.g. About, Contact) to ensure a polished and cohesive user experience.

- **`experimental/` directory**:  
  Contains exploratory scripts related to embedding creation, transcript downloading, and YouTube search experiments (e.g., `embedder.py`, `KNN.py`, `embed.py`).

- **`img/` directory**:  
  Stores images and icons used in the UI.

- **`data/` directory**:  
  Contains CSV files and other data resources. For instance, `ml_grouped_topics_questions.csv` is used to store and analyze student questions by topic.

## Key Features

1. **User Roles (Professor/Student)**:
   - **Professors**: Can create courses, upload materials, add YouTube links, and view data insights.
   - **Students**: Can view courses, ask questions, generate flashcards/MCQs, summarize content, and listen to podcasts generated from course materials.

2. **PDF to Podcast**:
   - Upload PDF materials.
   - Extract text using `PyPDF2`.
   - Generate a podcast script using GPT-4 via LangChain’s `LLMChain`.
   - Convert the script into an MP3 audio file with `gTTS`.
   - Listen to the generated podcast in the web app.

3. **AI-Powered Q&A and Content Generation**:
   - **Vector Stores with FAISS**: Documents (PDFs, transcripts) are split into chunks and embedded. FAISS stores these embeddings for semantic retrieval.
   - **Academic Q&A**: Students’ questions are answered based on the vector store context. The model only uses provided documents, ensuring accurate and relevant responses.
   - **Summarization**, **Flashcards**, and **MCQs**: Automatically create study aids and summaries from the uploaded materials.

4. **YouTube Integration**:
   - Suggest refined search keywords using GPT-3.5.
   - Retrieve top YouTube videos via web scraping.
   - Download or generate transcripts (if AssemblyAI key is provided).
   - Embed transcripts and recommend the best video for a given query.

5. **Data Insights and Visualization**:
   - Classify and store student questions by ML topics.
   - Visualize data in pie charts, bar charts, and word clouds.
   - Generate a detailed report (via GPT-4) on common question topics and insights to improve the course.

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```

2. **Install Dependencies**:
   It’s recommended to use a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

   Required Python packages include:
   - `streamlit`
   - `PyPDF2`
   - `langchain`
   - `gTTS`
   - `OpenAI` Python bindings
   - `SQLAlchemy`
   - `SentenceTransformers`
   - `FAISS`
   - `AssemblyAI` Python bindings
   - `yt-dlp` 
   - `BeautifulSoup4`, `urllib3` for web scraping
3. **Modify the .env file**  
   ```

4. **Run the App**:
   Start the Streamlit application:
   ```bash
   streamlit run application.py
   ```
Here’s an improved, well-organized version of your Docker notes:

---

### **Notes for Docker: Building and Running a Streamlit App**

#### **1. If you are using an M2 Mac or any ARM-based laptop:**
   - **Build**:
     ```bash
     docker build --platform linux/amd64 -t my-streamlit-app .
     ```
   - **Run**:
     ```bash
     docker run --env-file .env -p 8501:8501 my-streamlit-app
     ```

#### **2. If you are using any other system (e.g., AMD or Intel-based):**
   - Use `sudo` if required.
   - **Build**:
     ```bash
     docker build -t my-streamlit-app .
     ```
   - **Run**:
     ```bash
     docker run --env-file .env -p 8501:8501 my-streamlit-app
     ```

---

### **Accessing Your Streamlit App**
By default, your Streamlit app will run at:

   - **URL**: [http://localhost:8501](http://localhost:8501)

Simply open this link in your browser to view the app.

---
## Usage

1. **Home Page**:  
   From the homepage, you can navigate to sign up, log in, or explore About/Contact pages.

2. **Sign Up / Login**:  
   - **Professor**: After logging in, you’re taken to a dashboard where you can create courses, upload files, set YouTube links, and view insights.
   - **Student**: After logging in, you see available courses, can open them, ask questions, generate flashcards/MCQs, summarize materials, or convert them into podcasts.

3. **Course Management** (Professor Only):
   - Create a new course.
   - Upload PDFs/TXT files as course materials.
   - Add a YouTube video link, which will automatically process transcripts (if configured).

4. **Student Interactions**:
   - **View Course Details**: See what materials are available.
   - **Ask Questions**: The system uses vector embedding and GPT to answer.
   - **Generate Flashcards/MCQs**: Click the respective buttons and receive AI-generated study aids.
   - **Summarize Content**: Quickly get a summary of all course materials.
   - **Podcast Generation**: Upload PDFs and generate a script/audio podcast to study on the go.

5. **Data Insights** (Professor Only):
   - View charts and a report of all asked questions (topics distribution, word clouds).
   - Improve course content based on student query analysis.

## Customization

- **Models**:  
  You can switch GPT models (like `gpt-4` or `gpt-3.5-turbo`) by editing the code in `application.py` or `podcast.py`.
  
- **Styling**:  
  Custom CSS is injected at runtime. To adjust styling, modify `application1.py` or the inline CSS sections in the code.

- **Database**:  
  The default SQLite database can be swapped with another SQL database by updating `DATABASE_URL` in `database.py`.

- **Embeddings & Vector Store**:
  By default, it uses the `all-MiniLM-L6-v2` SentenceTransformer model. You can change the embedding model in `application.py` or other embedding scripts.


## Contributing

1. **Fork the Repo** and create feature branches.
2. **Pull Requests**: Submit PRs for review before merging.
3. **Issues**: Report bugs or request features via the Issues tab.
