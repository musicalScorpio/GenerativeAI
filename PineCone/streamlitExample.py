from io import StringIO

import pandas as pd
import requests
import streamlit as st
from fastapi import FastAPI
from transformers import pipeline

'''
app = FastAPI()
#uvicorn main:app --reload
@app.get("/{token}")
def read_root(token):
    return {"Seniment is ": getSentimentAnalysis(token)}

@app.get("/similar_questions/{token}")
def read_root(token):
    return {getSentimentAnalysis(token)}

'''
# streamlit run "/Users/samukhe/CODING/PYTHON/Generative AI/STEAMLIT/streamlitExample.py"


#result = st.button("Click here")
#if result:
#    st.write(":smile:")


#Senitement Analysis
txt = st.text_area('Text to analyze', '''
    It was the best of times, it was the worst of times, it was
    the age of wisdom, it was the age of foolishness, it was
    the epoch of belief, it was the epoch of incredulity, it
    was the season of Light, it was the season of Darkness, it
    was the spring of hope, it was the winter of despair, (...)
    ''')

#Similar Questions
txt_s_q = st.text_area('Type your question to get similar question ..')


def getSimilarQuestions(txt):
    data = requests.get(f"http://127.0.0.1:8000/similar_questions/{txt}").json()
    return data

st.write(getSimilarQuestions(txt_s_q))


def getSentimentAnalysis (input):
    generator = pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')
    return generator(input)

st.write('Sentiment:', getSentimentAnalysis(txt) )

st.title('This is just a Sample App..')
uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

def getCorrectURL (keyToLookUp):
    urlmap = {}
    urlmap["SENTIMENT_ANALYSIS"] = 'https://chat.openai.com/backend-api/conversation'
    '''
    {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]
    }
    
    '''
    urlmap["GENERATE_IMAGE"] = 'https://api.openai.com/v1/images/generations'
    '''
    {
        "prompt": "A cute baby sea otter",
        "n": 2,
        "size": "1024x1024"
    }
    '''
    urlmap["GENERATE_AUDIO_TRANSCRIPTION"] = 'https://api.openai.com/v1/audio/transcriptions'
    '''
    {
        "file": "audio.mp3",
        "model": "whisper-1"
    }
    '''
    return urlmap.get(keyToLookUp)


def sentimentAnalysisUsignAPI(text):
    key='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'
    headers={}
    headers["Authorization"] = f"Bearer {key}"
    headers["OpenAI-Organization"] ="org-AhI65UBIYE8XyUsOJDKdqzqu"
    x = requests.post(getCorrectURL("SENTIMENT_ANALYSIS"), data=text, headers=headers)






