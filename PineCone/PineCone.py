import os
import pinecone
from pinecone import GRPCIndex
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import os
#pip install -qU "pinecone-client[grpc]==2.2.1" datasets==2.12.0 sentence-transformers==2.2.2

"""
@author Sam Mukherjee
This is some sample code form PineCode website to test out the product. Use this to get started quickly for pocs.
"""
class PineCone:
    INDEX_NAME = 'semantic-search'
    BATCH_SIZE =128
    pinecone_index:GRPCIndex = None
    #Boiler plate Stuff
    def __init__(self, model:SentenceTransformer):
        print('Inside the PineCone......')
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
        if PineCone.INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(name= PineCone.INDEX_NAME, dimension=model.get_sentence_embedding_dimension(), metric='cosine')
        #Now connect to the Index
        PineCone.pinecone_index = pinecone.GRPCIndex(PineCone.INDEX_NAME)
    #Insert items as Vectors
    def update_or_insert(self, model:SentenceTransformer, questions:[]):
        #batch size = 128
        batch_size = PineCone.BATCH_SIZE
        #try:
        for i in tqdm ( range (0, len(questions) ,batch_size)):
            #find end of the batch
            i_end = min (i+batch_size , len(questions))
            #crate ids of the batch
            ids = [str (x) for x in range(i, i_end)]
            #create metadata batch
            metadatas = [{'text': text} for text in questions[i:i_end]]
            #create embeddigs
            xc = model.encode(questions[i:i_end])
            #create record list for upsert
            records = zip(ids,xc, metadatas)
            PineCone.pinecone_index.upsert(vectors=records)
        #except Exception:
           # print("Exception occured while connection to Pinecone.....")
        #test
        #check_no_of_recs
        print(PineCone.pinecone_index.describe_index_stats())

    def find_similar (self, query:str, model:SentenceTransformer):
        xq = model.encode(query).tolist()
        #Now query
        xc = PineCone.pinecone_index.query(xq, top_k=5, include_metadata=True)
        similar_questions = []
        for item in xc['matches']:
            str = item['metadata']['text']
            similar_questions.append(str)
        return similar_questions