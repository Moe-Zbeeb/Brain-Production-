import sys
import math
import bs4 as bs
import urllib.request
import re
import PyPDF2
import nltk
from nltk.stem import WordNetLemmatizer 
import spacy

# Execute this line if you are running this code for the first time
nltk.download('wordnet')

# Initializing variables
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

# Step 2. Define functions for Reading Input Text

# Function to Read .txt File and return its Text
def file_text(filepath):
    with open(filepath) as f:
        text = f.read().replace("\n", '')
    return text

# Function to Read PDF File and return its Text
def pdfReader(pdf_path):
    with open(pdf_path, 'rb') as pdfFileObject:
        pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
        count = pdfReader.numPages
        text = ""
        for i in range(count):
            page = pdfReader.getPage(i)
            text += page.extractText()
    return text

# Function to Read Wikipedia page URL and return its Text   
def wiki_text(url):
    scrap_data = urllib.request.urlopen(url)
    article = scrap_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = ""
    for p in paragraphs:
        article_text += p.text
    # Removing unwanted characters
    article_text = re.sub(r'\[[0-9]*\]', '', article_text)
    return article_text

# Step 3. Input Handling
# Set the input type here: "text", "file", "pdf", or "url"
input_type = "file"  # Change this to "text", "file", "pdf", or "url"

if input_type == "text":
    # Direct Text Input
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    """
elif input_type == "file":
    # Provide the file path to a .txt file
    txt_path =  "/home/mohammad/Brain-Production-/transcription.txt" # Change this to your file path
    text = file_text(txt_path)

elif input_type == "pdf":
    # Provide the file path to a .pdf file
    pdf_path = "example.pdf"  # Change this to your file path
    text = pdfReader(pdf_path)

elif input_type == "url":
    # Provide the URL to a Wikipedia article
    wiki_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"  # Change this to the URL you want
    text = wiki_text(wiki_url)

else:
    print("Invalid input type provided.")
    sys.exit()

# Step 4. Define functions to create Tf-Idf Matrix

# Function to calculate frequency of word in each sentence
def frequency_matrix(sentences):
    freq_matrix = {}
    stopWords = nlp.Defaults.stop_words
    for sent in sentences:
        freq_table = {}
        words = [word.text.lower() for word in sent if word.text.isalnum()]
        for word in words:  
            word = lemmatizer.lemmatize(word)
            if word not in stopWords:
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
        freq_matrix[sent[:15]] = freq_table
    return freq_matrix

# Function to calculate Term Frequency(TF) of each word
def tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        tf_table = {}
        total_words_in_sentence = len(freq_table)
        for word, count in freq_table.items():
            tf_table[word] = count / total_words_in_sentence
        tf_matrix[sent] = tf_table
    return tf_matrix

# Function to find how many sentences contain a 'word'
def sentences_per_words(freq_matrix):
    sent_per_words = {}
    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in sent_per_words:
                sent_per_words[word] += 1
            else:
                sent_per_words[word] = 1
    return sent_per_words

# Function to calculate Inverse Document frequency(IDF) for each word
def idf_matrix(freq_matrix, sent_per_words, total_sentences):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_sentences / float(sent_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix

# Function to calculate Tf-Idf score of each word
def tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, tf_value), (word2, idf_value) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(tf_value * idf_value)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

# Function to rate every sentence with some score calculated on basis of Tf-Idf
def score_sentences(tf_idf_matrix):
    sentenceScore = {}
    for sent, f_table in tf_idf_matrix.items():
        total_tfidf_score_per_sentence = 0
        total_words_in_sentence = len(f_table)
        for word, tf_idf_score in f_table.items():
            total_tfidf_score_per_sentence += tf_idf_score
        if total_words_in_sentence != 0:
            sentenceScore[sent] = total_tfidf_score_per_sentence / total_words_in_sentence
    return sentenceScore

# Function Calculating average sentence score 
def average_score(sentence_score):
    total_score = 0
    for sent in sentence_score:
        total_score += sentence_score[sent]
    average_sent_score = (total_score / len(sentence_score))
    return average_sent_score

# Function to return summary of article
def create_summary(sentences, sentence_score, threshold):
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentence_score and sentence_score[sentence[:15]] >= (threshold):
            summary += " " + sentence.text
    return summary

# Step 5. Using all functions to generate summary

# Counting number of words in original article
original_words = text.split()
original_words = [w for w in original_words if w.isalnum()]
num_words_in_original_text = len(original_words)

# Converting received text into spacy Doc object
text = nlp(text)

# Extracting all sentences from the text in a list
sentences = list(text.sents)
total_sentences = len(sentences)

# Generating Frequency Matrix
freq_matrix = frequency_matrix(sentences)

# Generating Term Frequency Matrix
tf_matrix = tf_matrix(freq_matrix)

# Getting number of sentences containing a particular word
num_sent_per_words = sentences_per_words(freq_matrix)

# Generating ID Frequency Matrix
idf_matrix = idf_matrix(freq_matrix, num_sent_per_words, total_sentences)

# Generating Tf-Idf Matrix
tf_idf_matrix = tf_idf_matrix(tf_matrix, idf_matrix)

# Generating Sentence score for each sentence
sentence_scores = score_sentences(tf_idf_matrix)

# Setting threshold to average value 
threshold = average_score(sentence_scores)

# Getting summary 
summary = create_summary(sentences, sentence_scores, 1.3 * threshold)

print("\n\n")
print("*"*20, "Summary", "*"*20)
print("\n")
print(summary)
print("\n\n")
print("Total words in original article = ", num_words_in_original_text)
print("Total words in summarized article = ", len(summary.split()))
