import os
import logging
import base64
import tempfile
from datetime import datetime

import PyPDF2
import streamlit as st
from sqlalchemy.orm import sessionmaker
from database import SessionLocal  # Ensure you have a 'database.py' with SessionLocal
from models import User, Course, CourseFile, StudentQuestion  # Ensure you have a 'models.py' with these classes
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain import LLMChain, PromptTemplate
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
from langchain.prompts import ChatPromptTemplate
from gtts import gTTS  # gTTS for text-to-speech

# ---------------------- Configuration ----------------------

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize in-memory cache
cache = InMemoryCache()

# Initialize database session
session_db = SessionLocal()

# Fetch OpenAI API key from environment variables for security
OPENAI_API_KEY = "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"  # Ensure this environment variable is set

if not OPENAI_API_KEY:
    logging.error("OpenAI API key is not set.")
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize LLM
try:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
        request_timeout=60  # seconds
    )
    logging.info("Successfully connected to OpenAI LLM.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI LLM: {str(e)}")
    st.error(f"Failed to initialize OpenAI LLM: {str(e)}")
    st.stop()

# ---------------------- LangchainHandler Class ----------------------

class LangchainHandler:
    def __init__(self, llm):
        self.llm = llm

    def load_document(self, file_path):
        """
        Load a document (PDF or text) from the specified file path.
        :param file_path: Path to the document file.
        :return: List of Document objects.
        """
        try:
            logging.info(f"Loading document from: {file_path}")
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            raw_docs = loader.load()
            logging.info(f"Loaded {len(raw_docs)} documents.")

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            docs = text_splitter.split_documents(raw_docs)
            logging.info(f"Split the document into {len(docs)} chunks.")
            return docs
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            return []

    def create_vector_store(self, docs):
        """
        Create a FAISS vector store from the provided documents.
        :param docs: List of Document objects.
        :return: FAISS vector store.
        """
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = FAISS.from_documents(docs, embeddings)
            logging.info("Created vector store from documents.")
            return vector_store
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            return None

    def get_response(self, vector_store, question):
        """
        Get a response to the user's question using the vector store.
        :param vector_store: FAISS vector store.
        :param question: User's question string.
        :return: Response string.
        """
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever
            )
            response = qa_chain.run(question)
            logging.info("Generated response to user question.")
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Sorry, I couldn't process your question at the moment."

    def summarize_documents(self, documents):
        """
        Summarize the list of documents using the LLM.
        :param documents: List of Document objects.
        :return: Summary string.
        """
        try:
            # Use the summarize chain with map_reduce to handle large documents
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            summary = chain.run(documents)
            logging.info("Generated summary of course materials.")
            return summary.strip()
        except Exception as e:
            logging.error(f"Error during summarization: {str(e)}")
            return "Sorry, I couldn't summarize the course materials at the moment."

    def generate_mcq_questions(self, documents, num_questions=10):
        """
        Generate multiple-choice questions from the documents using the LLM.
        :param documents: List of Document objects.
        :param num_questions: Number of MCQs to generate.
        :return: String containing the MCQs.
        """
        try:
            # Combine all documents into one text
            combined_text = "\n\n".join([doc.page_content for doc in documents])

            # Define the prompt template
            template = """
            You are a teacher creating assessment materials.
            Based on the following text, generate {num_questions} multiple-choice questions.

            Text:
            {text}

            Remember to ensure that the questions are clear and the options are not misleading.
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
            return "Sorry, I couldn't generate MCQs from the course materials at the moment."

    def generate_flashcards(self, documents, num_flashcards=20):
        """
        Generate flashcards from the documents using the LLM.
        :param documents: List of Document objects.
        :param num_flashcards: Number of flashcards to generate.
        :return: String containing the flashcards.
        """
        try:
            # Combine all documents into one text
            combined_text = "\n\n".join([doc.page_content for doc in documents])

            # Define the prompt template
            template = """
            You are an expert teacher creating study materials for students.
            Based on the following text, generate {num_flashcards} flashcards.

            Each flashcard should be formatted as:
            Q: [Question]
            A: [Answer]

            The questions should cover key concepts, definitions, and important details.

            Text:
            {text}

            Remember to ensure that the questions are clear and concise, focusing on essential information.
            """

            prompt = PromptTemplate(
                input_variables=["num_flashcards", "text"],
                template=template
            )

            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            response = llm_chain.run(num_flashcards=num_flashcards, text=combined_text)
            logging.info("Generated flashcards from course materials.")
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating flashcards: {str(e)}")
            return "Sorry, I couldn't generate flashcards from the course materials at the moment."

    def generate_podcast_script(self, extracted_text, openai_api_key):
        """
        Generate a podcast script using LangChain and OpenAI.
        """
        try:
            # Define the prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert podcaster. Create a detailed and engaging podcast script based on the following content."),
                ("user", "{content}")
            ])

            # Create an LLM chain with the provided OpenAI API key
            langchain_llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                openai_api_key=openai_api_key
            )
            chain = LLMChain(llm=langchain_llm, prompt=prompt_template)

            # Run the chain
            script = chain.run(content=extracted_text)
            logging.info("Generated podcast script.")
            return script.strip()
        except Exception as e:
            logging.error(f"Error generating podcast script: {e}")
            return ""

    def generate_podcast_audio(self, script, output_filename):
        """
        Convert the podcast script to speech using gTTS.
        """
        try:
            tts = gTTS(text=script, lang='en')
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, output_filename)
            tts.save(audio_path)
            logging.info(f'Audio content saved to "{audio_path}"')
            return audio_path
        except Exception as e:
            logging.error(f"Error converting text to speech with gTTS: {e}")
            return ""

# Initialize LangchainHandler
langchain_handler = LangchainHandler(llm=llm)

# ---------------------- Helper Functions ----------------------

def inject_css():
    """
    Inject custom CSS styles into the Streamlit app for enhanced UI.
    """
    st.markdown(
        """
        <style>
        /* General Styling */
        body {
            background-color: #f0f2f6;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main {
            padding: 2rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #333333;
            font-weight: 600;
        }
        /* Button Styling */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
        }
        .stButton > button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        /* Input Box Styling */
        .stTextInput > div > div > input {
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #cccccc;
            font-size: 14px;
        }
        /* Radio Button Styling */
        .stRadio > div > label {
            font-size: 16px;
            color: #555555;
        }
        /* Expander Styling */
        .stExpander > div > div > div:first-child {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        /* Card Styling */
        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        /* Icons */
        .icon {
            font-size: 20px;
            margin-right: 10px;
            color: #4CAF50;
        }
        /* Visualization Container */
        .viz-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        /* Report Container */
        .report-container {
            background-color: #ffffff;
            color: #333333;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: auto;
            overflow-x: auto;
        }
        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def extract_text_from_pdf(pdf_file_path_or_object):
    """
    Extract text from a PDF file.
    """
    try:
        if isinstance(pdf_file_path_or_object, str):
            pdf_reader = PyPDF2.PdfReader(pdf_file_path_or_object)
        else:
            pdf_reader = PyPDF2.PdfReader(pdf_file_path_or_object)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

# ---------------------- Page Functions ----------------------

def welcome_page():
    """
    Displays the welcome page with a styled message and a button to navigate to the login/signup page.
    """
    inject_css()
    st.markdown(
        """
        <div class="main">
            <h1>Think <span style="color:#4CAF50;">Smarter,</span> Not Harder</h1>
            <h3>The ultimate tool for understanding the information that matters most to you.</h3>
            <br>
            <div style="text-align: center;">
                <form action="#" method="post">
                </form>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Handle button click
    if st.button("Continue to Login"):
        st.session_state.page = "login"

def login_page():
    """
    Displays the login and signup page.
    """
    inject_css()
    st.title("Welcome to Course Chat App")
    st.markdown("---")
    option = st.radio("Choose an option", ["Login", "Sign Up"], horizontal=True)

    if option == "Login":
        st.subheader("Login")
        with st.form(key='login_form'):
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            submit = st.form_submit_button("Login")
            if submit:
                if not username.strip() or not password:
                    st.error("Please enter both username and password.")
                    return
                user = session_db.query(User).filter_by(username=username.strip()).first()
                if user and user.check_password(password):
                    st.session_state.user = user
                    st.session_state.page = "dashboard"
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid username or password.")
    else:
        st.subheader("Sign Up")
        with st.form(key='signup_form'):
            new_username = st.text_input("Username", key='new_username')
            new_password = st.text_input("Password", type='password', key='new_password')
            confirm_password = st.text_input("Confirm Password", type='password', key='confirm_password')
            role = st.selectbox("Role", ['professor', 'student'])
            submit = st.form_submit_button("Sign Up")
            if submit:
                if not new_username.strip() or not new_password or not confirm_password:
                    st.error("Please fill out all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif session_db.query(User).filter_by(username=new_username.strip()).first():
                    st.error("Username already exists. Please choose another one.")
                else:
                    new_user = User(username=new_username.strip(), role=role)
                    new_user.set_password(new_password)
                    session_db.add(new_user)
                    session_db.commit()
                    st.success("Account created successfully! You can now log in.")
                    st.info("Please switch to the Login tab.")

def professor_page():
    """
    Displays the professor dashboard with options to create and manage courses.
    """
    inject_css()
    st.title("👨‍🏫 Professor Dashboard")
    st.markdown("---")
    with st.sidebar:
        st.header("Navigation")
        selected_tab = st.radio("Go to", ["Create Course", "Manage Courses"])
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state.user.username}")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "welcome"

    if selected_tab == "Create Course":
        create_course_section()
    else:
        manage_courses_section()

def create_course_section():
    """
    Allows professors to create a new course.
    """
    st.header("Create a New Course")
    with st.form(key='create_course_form'):
        course_name = st.text_input("Course Name")
        submit = st.form_submit_button("Create Course")
        if submit:
            if not course_name.strip():
                st.error("Course name cannot be empty.")
                return
            if session_db.query(Course).filter_by(name=course_name.strip()).first():
                st.error("A course with this name already exists.")
                return
            new_course = Course(name=course_name.strip(), professor_id=st.session_state.user.id)
            session_db.add(new_course)
            session_db.commit()
            st.success(f"Course '{course_name}' created successfully!")
            # Update the courses list in session state
            if 'courses' in st.session_state:
                st.session_state.courses.append(new_course)
            else:
                st.session_state.courses = [new_course]

def manage_courses_section():
    """
    Allows professors to manage their courses, including viewing the pie chart, bar chart, word cloud, report, and deleting courses.
    """
    st.header("Manage Your Courses")
    courses = session_db.query(Course).filter_by(professor_id=st.session_state.user.id).all()

    if not courses:
        st.info("You have not created any courses yet.")
        return

    # Define the path to your backend CSV file
    csv_file_path = "ml_grouped_topics_questions.csv"  # <-- UPDATE THIS PATH

    for course in courses:
        st.markdown("---")
        st.subheader(f"📚 {course.name}")
        st.markdown(f"### Course Actions")
        with st.container():
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### Upload Course Materials")
                with st.form(key=f'upload_form_{course.id}', clear_on_submit=True):
                    uploaded_files = st.file_uploader(
                        "Upload files (PDF or TXT)", accept_multiple_files=True,
                        key=f"upload_{course.id}"
                    )
                    submit = st.form_submit_button("Upload Files")
                    if submit:
                        if uploaded_files:
                            for uploaded_file in uploaded_files:
                                if uploaded_file.size > 10 * 1024 * 1024:  # 10 MB
                                    st.warning(f"File {uploaded_file.name} exceeds 10MB and was skipped.")
                                    continue
                                existing_file = session_db.query(CourseFile).filter_by(
                                    course_id=course.id, filename=uploaded_file.name).first()
                                if existing_file:
                                    st.warning(f"File {uploaded_file.name} already exists and was skipped.")
                                    continue
                                course_file = CourseFile(
                                    filename=uploaded_file.name,
                                    data=uploaded_file.read(),
                                    course_id=course.id
                                )
                                session_db.add(course_file)
                            session_db.commit()
                            st.success("Files uploaded successfully!")
                            # Update the course files in session state
                            course.files = session_db.query(CourseFile).filter_by(course_id=course.id).all()
                        else:
                            st.error("No files selected.")

                st.markdown("#### Current Files")
                if session_db.query(CourseFile).filter_by(course_id=course.id).count() > 0:
                    for file in course.files:
                        file_bytes = base64.b64encode(file.data).decode()
                        href = f'<a href="data:file/octet-stream;base64,{file_bytes}" download="{file.filename}">{file.filename}</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.info("No files uploaded for this course.")

            with col2:
                st.markdown("#### Course Insights")
                # Initialize session state variables for this course if not already set
                if f"show_insights_{course.id}" not in st.session_state:
                    st.session_state[f"show_insights_{course.id}"] = False

                if st.button("Toggle Insights", key=f"toggle_insights_{course.id}"):
                    st.session_state[f"show_insights_{course.id}"] = not st.session_state[f"show_insights_{course.id}"]

                # Clear All Button
                if st.button("Clear Insights", key=f"clear_insights_{course.id}"):
                    st.session_state[f"show_insights_{course.id}"] = False

        # Display Visualizations and Report
        if st.session_state[f"show_insights_{course.id}"]:
            st.markdown("### Course Insights")
            insights_container = st.container()
            with insights_container:
                if os.path.exists(csv_file_path):
                    df = pd.read_csv(csv_file_path)
                    if 'Topic' not in df.columns or 'Question' not in df.columns:
                        st.error("CSV must have 'Topic' and 'Question' columns.")
                        continue

                    tabs = st.tabs(["📊 Pie Chart", "📈 Bar Chart", "☁️ Word Cloud", "📄 Report"])
                    with tabs[0]:
                        pie_fig = generate_pie_chart(df)
                        st.plotly_chart(pie_fig, use_container_width=True)

                    with tabs[1]:
                        bar_fig = generate_bar_chart(df)
                        st.plotly_chart(bar_fig, use_container_width=True)

                    with tabs[2]:
                        wordcloud_img = generate_wordcloud(df)
                        st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True)

                    with tabs[3]:
                        report = generate_csv_report(csv_file_path)
                        if report.startswith("Error generating report"):
                            st.error(report)
                        else:
                            st.markdown(report, unsafe_allow_html=True)
                else:
                    st.error(f"CSV file not found at the specified path: {csv_file_path}")

        # Delete Course
        with st.container():
            st.markdown("### Delete Course")
            with st.form(key=f'delete_course_form_{course.id}', clear_on_submit=True):
                confirm = st.checkbox("Are you sure you want to delete this course? This action cannot be undone.", key=f"confirm_delete_{course.id}")
                submit = st.form_submit_button("Delete Course")
                if submit:
                    if confirm:
                        for file in course.files:
                            session_db.delete(file)
                        session_db.delete(course)
                        session_db.commit()
                        st.success(f"Course '{course.name}' deleted successfully!")
                        # Refresh the page or update session state as needed
                        st.experimental_rerun()
                    else:
                        st.error("Please confirm to delete the course.")

def generate_pie_chart(df):
    """
    Generate an interactive pie chart using Plotly.
    :param df: DataFrame containing the 'Topic' column.
    :return: Plotly figure.
    """
    topic_counts = df['Topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig = px.pie(topic_counts, names='Topic', values='Count', title='Topic Distribution',
                 hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def generate_bar_chart(df):
    """
    Generate an interactive horizontal bar chart using Plotly.
    :param df: DataFrame containing the 'Topic' column.
    :return: Plotly figure.
    """
    topic_counts = df['Topic'].value_counts().sort_values(ascending=True).reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig = px.bar(topic_counts, x='Count', y='Topic', orientation='h',
                 title='Questions per Topic', color='Count', color_continuous_scale='Blues')
    fig.update_layout(showlegend=False)
    return fig

def generate_wordcloud(df):
    """
    Generate a word cloud image.
    :param df: DataFrame containing the 'Question' column.
    :return: Base64 encoded image string.
    """
    text = " ".join(df['Question'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          min_font_size=10).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud', fontsize=16)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    encoded = base64.b64encode(img_bytes).decode()
    plt.close(fig)
    return encoded

def generate_csv_report(csv_file_path):
    """
    Generate a detailed report from the CSV file using the LLM.
    :param csv_file_path: Path to the CSV file.
    :return: String containing the report.
    """
    try:
        df = pd.read_csv(csv_file_path)
        # Remove any PII or sensitive information if present
        # For example, if there is a 'StudentID' column:
        # df = df.drop(columns=['StudentID'], errors='ignore')
        
        # Sample data if too large
        max_rows = 500  # Adjust based on the token limit (you may need to experiment)
        if len(df) > max_rows:
            df_sample = df.sample(n=max_rows, random_state=42)
        else:
            df_sample = df

        # Convert the DataFrame to a CSV string
        csv_data = df_sample.to_csv(index=False)
        
        # Define the prompt template
        template = """
        You are a data analyst assisting a professor in understanding student questions from a course.
        Based on the following CSV data, generate a detailed and rich report that includes:

        - An overview of the total number of questions.
        - The number of unique topics covered.
        - Insights into the most common topics.
        - Any noticeable trends or patterns.
        - Suggestions for areas that may need more focus based on the questions.

        CSV Data:
        {csv_data}

        Please present the report in a clear and organized manner, using headings and bullet points where appropriate.
        """
        
        prompt = PromptTemplate(
            input_variables=["csv_data"],
            template=template
        )
        
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        report = llm_chain.run(csv_data=csv_data)
        logging.info("Generated detailed report using LLM.")
        return report.strip()
    except Exception as e:
        logging.error(f"Error generating detailed report: {str(e)}")
        return f"Error generating report: {e}"

def student_page():
    """
    Displays the student dashboard with podcast generation feature within each course's expander.
    """
    inject_css()
    st.title("👩‍🎓 Student Dashboard")
    st.markdown("---")
    
    # ------------------ Available Courses Section ------------------
    st.header("Available Courses")
    courses = session_db.query(Course).all()

    if not courses:
        st.info("No courses available at the moment.")
        return

    for course in courses:
        with st.expander(f"📘 {course.name}", expanded=False):
            # Display Files
            st.subheader("📁 Course Materials")
            if session_db.query(CourseFile).filter_by(course_id=course.id).count() > 0:
                for file in course.files:
                    file_bytes = base64.b64encode(file.data).decode()
                    href = f'<a href="data:file/octet-stream;base64,{file_bytes}" download="{file.filename}">{file.filename}</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No materials uploaded for this course.")

            # ------------------ Generate Podcast Feature ------------------
            generate_podcast_for_course(course, OPENAI_API_KEY)
            st.markdown("---")

            # Chat with Course Material
            st.subheader("💬 Chat with Course Material")
            with st.form(key=f'chat_form_{course.id}', clear_on_submit=True):
                user_question = st.text_input(f"Ask a question about {course.name}:", key=f"question_input_{course.id}")
                submit = st.form_submit_button("Send")
                if submit:
                    if user_question.strip():
                        with st.spinner("Processing your question..."):
                            try:
                                response = chat_with_documents(course, user_question)
                                st.success("Response:")
                                st.write(response)
                            except Exception as e:
                                st.error(f"An error occurred: {e}")
                    else:
                        st.error("Please enter a question.")

            # Study with Flashcards
            st.subheader("📚 Study with Flashcards")
            with st.form(key=f'flashcards_form_{course.id}', clear_on_submit=True):
                submit = st.form_submit_button("Generate Flashcards")
                if submit:
                    with st.spinner("Generating flashcards..."):
                        try:
                            flashcards = generate_flashcards_for_course(course)
                            st.success("🃏 Here are your flashcards:")
                            st.write(flashcards)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

            # Assess Your Knowledge (MCQs)
            st.subheader("📝 Assess Your Knowledge")
            with st.form(key=f'mcq_form_{course.id}', clear_on_submit=True):
                submit = st.form_submit_button("Generate MCQs")
                if submit:
                    with st.spinner("Generating MCQs..."):
                        try:
                            mcqs = generate_mcq_for_course(course)
                            st.success("🔍 Multiple-Choice Questions:")
                            st.write(mcqs)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

            # Summarize Course
            st.subheader("📄 Summarize Course")
            with st.form(key=f'summarize_form_{course.id}', clear_on_submit=True):
                submit = st.form_submit_button("Get Summary")
                if submit:
                    with st.spinner("Generating summary..."):
                        try:
                            summary = summarize_course_documents(course)
                            st.success("📖 Course Summary:")
                            st.write(summary)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

def generate_podcast_for_course(course, openai_api_key):
    """
    Allows students to generate a podcast based on the course materials or by uploading additional PDFs.
    """
    # Initialize session state for podcast if not already set
    podcast_audio_key = f"podcast_audio_{course.id}"
    script_key = f"script_{course.id}"
    if podcast_audio_key not in st.session_state:
        st.session_state[podcast_audio_key] = ""
    if script_key not in st.session_state:
        st.session_state[script_key] = ""

    st.markdown("### 🎙 Generate Podcast for This Course")

    # Create a container for better layout management
    with st.container():
        # Upload PDF Files Section
        st.markdown("#### 📁 Upload Additional PDF File(s) for Podcast")
        uploaded_files = st.file_uploader(
            f"📁 Upload PDF File(s) for {course.name}",
            accept_multiple_files=True,
            type=['pdf'],
            key=f"podcast_upload_{course.id}"
        )

        # Generate Podcast Button
        generate_btn = st.button("🎬 Generate Podcast", key=f"generate_podcast_{course.id}")

        if generate_btn:
            if not uploaded_files:
                st.error("Please upload at least one PDF file to generate a podcast.")
                return

            all_text = ""
            for uploaded_file in uploaded_files:
                st.info(f"Processing file: {uploaded_file.name}")
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    all_text += extracted_text + "\n"
                    st.success(f"Extracted text from {uploaded_file.name}")
                else:
                    st.warning(f"No text extracted from {uploaded_file.name}.")

            if all_text:
                # Generate Podcast Script
                st.info("Generating podcast script...")
                script = langchain_handler.generate_podcast_script(all_text, openai_api_key)
                if script:
                    st.session_state[script_key] = script
                    st.success("Podcast script generated successfully!")

                    # Display Script Optionally
                    if st.checkbox("📝 View Generated Script", key=f"view_script_{course.id}"):
                        st.markdown("### Generated Podcast Script")
                        st.write(script)

                    # Convert Script to Audio
                    st.info("Converting script to audio...")
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    output_filename = f"podcast_{course.id}_{timestamp}.mp3"
                    podcast_audio_path = langchain_handler.generate_podcast_audio(script, output_filename)

                    if podcast_audio_path:
                        st.session_state[podcast_audio_key] = podcast_audio_path
                        st.success("Audio podcast generated successfully!")

                        # Play Audio
                        st.markdown("### 🎧 Listen to Your Podcast")
                        st.audio(podcast_audio_path, format='audio/mp3')
                    else:
                        st.error("Failed to convert script to audio.")
                else:
                    st.error("Failed to generate podcast script.")
            else:
                st.error("No text extracted from the uploaded files.")

def chat_with_documents(course, question):
    """
    Load the course documents, create a vector store, get a response using RAG, and store the student's question.
    :param course: Course object containing the files.
    :param question: User's question string.
    :return: Response string from OpenAI.
    """
    documents = []
    for file in course.files:
        try:
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
                tmp_file.write(file.data)
                tmp_file_path = tmp_file.name
            # Load the document using LangchainHandler
            docs = langchain_handler.load_document(tmp_file_path)
            if docs:
                documents.extend(docs)
        except Exception as e:
            st.warning(f"Could not process file {file.filename}: {type(e).__name__}: {e}")
            continue

    if not documents:
        raise ValueError("No readable course materials available.")

    # Create vector store
    vector_store = langchain_handler.create_vector_store(documents)
    if not vector_store:
        raise ValueError("Failed to create vector store from documents.")

    # Get response
    response = langchain_handler.get_response(vector_store, question)

    # Store the student's question
    if st.session_state.user and st.session_state.user.role == 'student':
        try:
            new_question = StudentQuestion(
                user_id=st.session_state.user.id,
                course_id=course.id,
                question=question
            )
            session_db.add(new_question)
            session_db.commit()
            logging.info(f"Stored question from user {st.session_state.user.username} for course {course.name}.")
        except Exception as e:
            session_db.rollback()
            logging.error(f"Error storing student question: {str(e)}")
            st.error("An error occurred while saving your question.")

    return response

def summarize_course_documents(course):
    """
    Generate a summary of the course materials.
    :param course: Course object containing the files.
    :return: Summary string.
    """
    documents = []
    for file in course.files:
        try:
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
                tmp_file.write(file.data)
                tmp_file_path = tmp_file.name
            # Load the document using LangchainHandler
            docs = langchain_handler.load_document(tmp_file_path)
            if docs:
                documents.extend(docs)
        except Exception as e:
            st.warning(f"Could not process file {file.filename}: {type(e).__name__}: {e}")
            continue

    if not documents:
        raise ValueError("No readable course materials available.")

    # Generate summary using LangChainHandler
    summary = langchain_handler.summarize_documents(documents)
    return summary

def generate_mcq_for_course(course):
    """
    Generate MCQs from the course materials.
    :param course: Course object containing the files.
    :return: String containing the MCQs.
    """
    documents = []
    for file in course.files:
        try:
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
                tmp_file.write(file.data)
                tmp_file_path = tmp_file.name
            # Load the document using LangchainHandler
            docs = langchain_handler.load_document(tmp_file_path)
            if docs:
                documents.extend(docs)
        except Exception as e:
            st.warning(f"Could not process file {file.filename}: {type(e).__name__}: {e}")
            continue

    if not documents:
        raise ValueError("No readable course materials available.")

    # Generate MCQs using LangChainHandler
    mcqs = langchain_handler.generate_mcq_questions(documents)
    return mcqs

def generate_flashcards_for_course(course):
    """
    Generate flashcards from the course materials.
    :param course: Course object containing the files.
    :return: String containing the flashcards.
    """
    documents = []
    for file in course.files:
        try:
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
                tmp_file.write(file.data)
                tmp_file_path = tmp_file.name
            # Load the document using LangchainHandler
            docs = langchain_handler.load_document(tmp_file_path)
            if docs:
                documents.extend(docs)
        except Exception as e:
            st.warning(f"Could not process file {file.filename}: {type(e).__name__}: {e}")
            continue

    if not documents:
        raise ValueError("No readable course materials available.")

    # Generate flashcards using LangChainHandler
    flashcards = langchain_handler.generate_flashcards(documents)
    return flashcards

# ---------------------- Main Function ----------------------

def main():
    """
    Main function to control the flow of the Streamlit app.
    """
    # Set page configuration once at the top
    st.set_page_config(page_title="Course Chat App", layout="wide", initial_sidebar_state="expanded")

    # Initialize session state for user and page
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"

    # Page Navigation
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "dashboard":
        if st.session_state.user.role == 'professor':
            professor_page()
        else:
            student_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
