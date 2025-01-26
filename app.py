import re
import os
import nltk
import spacy
import textract
import pandas as pd
import streamlit as st
import pickle as pk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS, WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import chardet
from docx import Document

# Load spaCy model and other files
nlp = spacy.load('en_core_web_sm')
model = pk.load(open('modelDT.pkl', 'rb'))
Vectorizer = pk.load(open('vector.pkl', 'rb'))

# Skill categories (including more specific Workday skills)
category_skills = {
    "SQL Developer": ['sql', 'database', 'queries', 'etl', 'mysql', 'postgres', 'relational', 'ssis'],
    "React JS Developer": ['react', 'redux', 'javascript', 'html', 'css', 'node.js', 'frontend', 'web'],
    "Workday": [
        'workday hcm', 'staffing', 'compensation', 'eib', 'business process', 'calculated fields', 
        'report writer', 'domain security', 'security groups', 'workday security', 'functional area', 'workday integration'
    ],
    "PeopleSoft": ['peoplesoft', 'oracle', 'hrms', 'payroll', 'sap', 'erp', 'hr', 'peopletools', 'peoplesoft hr'],
}

# Streamlit title
st.title('Resume Classification App')
st.subheader('Upload your resume for automatic classification')

# Extract skills function
def extract_skills(resume_text):
    tokens = [token.text.lower() for token in nlp(resume_text) if not token.is_stop]
    data = pd.read_csv('skills.csv')  # Assuming this CSV contains relevant skills
    skills = list(data.columns.values)
    skillset = [token for token in tokens if token in skills]
    return list(set(skillset))

# Preprocessing function
def preprocess(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub(r'{html}', "", sentence)
    sentence = re.sub(r'<.*?>', '', sentence)
    sentence = re.sub(r'http\S+', '', sentence)
    sentence = re.sub(r'[0-9]+', '', sentence)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [word for word in tokens if len(word) > 2 and word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return " ".join(lemmatized_words)

# Extract text from DOCX file
def get_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Streamlit file uploader
uploaded_files = st.file_uploader("Upload Resume (DOCX Only)", type=['docx'], accept_multiple_files=True, key="resume_uploader")

# DataFrame to store results
file_data = pd.DataFrame(columns=['File Name', 'Predicted Category', 'Extracted Skills'])

# Process files
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Extract and preprocess text
        file_text = get_text_from_docx(uploaded_file)
        cleaned_text = preprocess(file_text)
        
        # Predict category using the model
        prediction_proba = model.predict_proba(Vectorizer.transform([cleaned_text]))[0]
        predicted_category = model.classes_[prediction_proba.argmax()]
        max_prob = max(prediction_proba)

        # Extract skills from the resume
        skills = extract_skills(file_text)

        # Check if Workday-specific keywords are present in the entire resume text
        workday_keywords = ['workday hcm', 'staffing', 'compensation', 'eib', 'business process', 
                            'calculated fields', 'report writer', 'domain security', 'security groups', 
                            'workday security', 'functional area', 'workday integration']
        
        # Check for Workday keywords in the full resume text
        workday_matches = [keyword for keyword in workday_keywords if keyword in file_text.lower()]

        # If any Workday keywords are found, classify as Workday
        if workday_matches:
            best_category = 'Workday'
        else:
            # If no direct Workday matches, prioritize by skill category matching
            detected_categories = {
                category: len([skill for skill in skills if skill in keywords])
                for category, keywords in category_skills.items()
            }
            best_category = max(detected_categories, key=detected_categories.get, default=predicted_category)

        # Store the results in the DataFrame
        file_data = pd.concat([file_data, pd.DataFrame([{
            'File Name': uploaded_file.name,
            'Predicted Category': best_category,
            'Extracted Skills': ", ".join(skills)
        }])], ignore_index=True)

    # Display results in Streamlit
    st.write(file_data)
    # Filtering by predicted category
    category_filter = st.selectbox("Filter by Category", file_data['Predicted Category'].unique())
    filtered_data = file_data[file_data['Predicted Category'] == category_filter]
    st.table(filtered_data)

    # Visualize skills in Word Cloud
    text_for_wordcloud = " ".join(file_data['Extracted Skills'])
    wordcloud = WordCloud(width=1000, height=800, background_color='white', stopwords=STOPWORDS).generate(text_for_wordcloud)
    st.subheader('Word Cloud of Extracted Skills')
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
