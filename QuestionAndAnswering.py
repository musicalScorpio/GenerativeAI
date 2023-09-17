"""
@author Sam Mukherjee
Step 1 : Ask for a file.
Step 2 : Chunk up the file.
Step 3: Store the chunks and get embeddings.
Step 4: Store the embeddings in a Vector DB.
Step 5: Ask for a question.
    Step 5.1: Do an similar search for least cosine distance.
    Step 5.2 : Use those chunks to create an well articulated answer.
Note: You will need your own OpenAI key
"""

from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from functools import lru_cache

load_dotenv('key.env')
prompt_template = """You are a helpful assistant, answer the question with the following text below within triple backticks
            ```
            {CONTEXT}
            ```
            Question : {QUESTION}
            If you find the information in the context do not make up an answer, just say NA.
            """


"""
We get the pdf and then chunk it up.
Ref :https://python.langchain.com/docs/modules/data_connection/document_transformers/
Ref: https://python.langchain.com/docs/use_cases/question_answering/
"""
@lru_cache()
def get_splits(chunksize:int=500, chunkoverlap:int=50)-> list:
    file = input("Type the PDF file path")
    pdfLoader = PyPDFLoader(file_path=file)
    pdf = pdfLoader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
    texts = text_splitter.split_documents(pdf)
    return texts

def get_output_from_llm(question:str):
    embeddings = OpenAIEmbeddings()
    prompt = PromptTemplate(template=prompt_template, input_variables=["CONTEXT", "QUESTION"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    docs = get_splits()
    vectorstore = Chroma.from_documents(docs, embeddings)
    #Retrive the results as part of the vector store. Vector store can be swapped by other vector stores.
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=vectorstore.as_retriever())
    result = qa_chain({"question": question})
    print(result)


if __name__ == "__main__":
    get_output_from_llm("How long should I wait after a fill charge ?")