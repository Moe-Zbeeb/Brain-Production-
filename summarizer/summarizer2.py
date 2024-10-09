import os
from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, TextLoader  
import re
os.environ['OPENAI_API_KEY'] = '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'

# Set your OpenAI API key (you can use environment variables or specify it directly)
openai_api_key = os.getenv("OPENAI_API_KEY")  # Or replace with "your_openai_api_key"

llm = OpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
text_splitter = CharacterTextSplitter()

# Function to remove timestamps in the format [xxx.xx - xxx.xx]
def remove_timestamps(text):
    return re.sub(r'\[\d{1,3}\.\d{2} - \d{1,3}\.\d{2}\]', '', text)

# Function to load the file (PDF or TXT)
def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
        docs = loader.load_and_split()
    else:
        raise ValueError("Unsupported file format. Please upload a .pdf or .txt file.")
    return docs

# Function to summarize the file
def summarize_file(file_path, custom_prompt=""):
    docs = load_file(file_path)
    
    # Clean documents to remove timestamps before summarizing
    for doc in docs:
        doc.page_content = remove_timestamps(doc.page_content)

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    
    # If there's a custom prompt, generate a custom summary
    if custom_prompt:
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                     map_prompt=PROMPT, combine_prompt=PROMPT)
        custom_summary = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        custom_summary = remove_timestamps(custom_summary)
    else:
        custom_summary = ""
    
    # Remove any remaining timestamps from the standard summary
    summary = remove_timestamps(summary)
    
    return summary, custom_summary

# Main function to set file paths and custom prompt
def main(filer):
    # Define the paths and custom prompt directly in the script
    file_path = filer  # Change this to the path of your .pdf or .txt file
    custom_prompt = "Provide 100 lines with a detailed overview of the document, excluding timestamps, and review the lecture. Give a detailed summary with enough depth that the student can review only from this summary."  # Optional custom prompt

    # Summarize the file
    summary, custom_summary = summarize_file(file_path, custom_prompt)

    # Print the results
    print("\nStandard Summary:")
    print(summary)
    
    if custom_summary:
        print("\nCustom Summary (based on your custom prompt):")
        print(custom_summary)

if __name__ == "__main__":
    main()