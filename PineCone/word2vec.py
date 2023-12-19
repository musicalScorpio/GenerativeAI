# Python program to generate word vectors using Word2Vec

# importing all necessary modules
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')

import nltk
'''
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
'''
alice_in_wonderland = open('alice_in_wonderland.txt')
text = alice_in_wonderland.read()
text = text.replace('\n', ' ')

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 256,
    chunk_overlap  = 20
)
docs = text_splitter.create_documents([text])
print(docs)

from langchain.text_splitter import NLTKTextSplitter
text_splitter = NLTKTextSplitter()
docs = text_splitter.split_text(text)
print(docs)


import gensim
from gensim.models import Word2Vec

#Rad the tile and then create vectors


data =[]

for i in sent_tokenize(text):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)
# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count=1,  vector_size=100, window=5)
print("Cosine similarity between 'alice' " + "and 'hurt' - CBOW : ", model1.wv.similarity('alice', 'hurt'))

