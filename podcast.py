import os
import logging
import tempfile
from datetime import datetime

import PyPDF2
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from gtts import gTTS  # gTTS for text-to-speech

# ---------------------- Configuration ----------------------

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Accepted file types for upload
ACCEPTED_FILE_INPUTS = ['pdf']

# Hardcoded OpenAI API key (replace with your actual key)
OPENAI_API_KEY = "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"

# ---------------------- Helper Functions ----------------------

def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def generate_podcast_script(extracted_text):
    """
    Generate a podcast script using LangChain and OpenAI.
    """
    try:
        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert podcaster. Create a detailed and engaging podcast script based on the following content."),
            ("user", "{content}")
        ])

        # Create an LLM chain with a hardcoded OpenAI API key
        langchain_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )
        chain = LLMChain(llm=langchain_llm, prompt=prompt_template)

        # Run the chain
        script = chain.run(content=extracted_text)
        return script.strip()

    except Exception as e:
        logger.error(f"Error generating podcast script: {e}")
        return ""

def convert_text_to_speech_gtts(script, output_filename):
    """
    Convert the podcast script to speech using gTTS.
    """
    try:
        tts = gTTS(text=script, lang='en')
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, output_filename)
        tts.save(audio_path)
        logger.info(f'Audio content saved to "{audio_path}"')
        return audio_path
    except Exception as e:
        logger.error(f"Error converting text to speech with gTTS: {e}")
        return ""

# ---------------------- Streamlit App ----------------------

def main():
    # Initialize session state
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'podcast_audio' not in st.session_state:
        st.session_state.podcast_audio = ""
    if 'script' not in st.session_state:
        st.session_state.script = ""

    # App Title
    st.set_page_config(
        page_title="PodfAI",
        page_icon="🎙",
    )
    st.title("🎙 PodfAI")
    st.markdown("### Generate podcast-style content based on your PDF input.")

    # File Uploader
    uploaded_files = st.file_uploader(
        "📁 Upload PDF File(s)",
        accept_multiple_files=True,
        type=ACCEPTED_FILE_INPUTS,
    )

    # Generate Button
    generate_btn = st.button("🎬 Generate Podcast")

    if generate_btn:
        if uploaded_files:
            all_text = ""
            for uploaded_file in uploaded_files:
                st.info(f"Processing file: {uploaded_file.name}")
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    all_text += extracted_text + "\n"
                else:
                    st.warning(f"No text extracted from {uploaded_file.name}.")

            if all_text:
                # Generate Podcast Script
                st.info("Generating podcast script...")
                script = generate_podcast_script(all_text)
                if script:
                    st.session_state.script = script
                    st.success("Podcast script generated successfully!")

                    # Display Script Optionally
                    if st.checkbox("📝 View Generated Script"):
                        st.markdown("### Generated Podcast Script")
                        st.write(script)

                    # Convert Script to Audio
                    st.info("Converting script to audio...")
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    output_filename = f"podcast_{timestamp}.mp3"
                    podcast_audio_path = convert_text_to_speech_gtts(script, output_filename)

                    if podcast_audio_path:
                        st.session_state.podcast_audio = podcast_audio_path
                        st.success("Audio podcast generated successfully!")

                        # Play Audio
                        st.markdown("### 🎧 Listen to Your Podcast")
                        st.audio(podcast_audio_path, format='audio/mp3')

                        # Display Transcript
                        st.markdown("### 📝 Transcript")
                        st.write(all_text)

                    else:
                        st.error("Failed to convert script to audio.")
                else:
                    st.error("Failed to generate podcast script.")
            else:
                st.error("No text extracted from the uploaded files.")
        else:
            st.error("No file uploaded. Please upload a PDF file to generate a podcast.")

if __name__ == "__main__":
    main()