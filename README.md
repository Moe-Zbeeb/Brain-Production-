

# Docker: Building and Running

## 1. General Instructions

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t my-streamlit-app .
```

### **Run the Docker Container**
```bash
docker run --env-file .env -p 8501:8501 my-streamlit-app
```

---

## **2. Instructions for Linux**

- Use `sudo` if required.

### **Build the Docker Image**
```bash
sudo docker build -t my-streamlit-app .
```

### **Run the Docker Container**
```bash
sudo docker run --env-file .env -p 8501:8501 my-streamlit-app
```


## Overview
![image](https://github.com/user-attachments/assets/9f6348f2-c648-4023-aac2-55e283d6ed06)




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
