from datasets import load_dataset
from PineCone import PineCone

from dotenv import load_dotenv

load_dotenv('../key.env')
"""
@author Sam Mukherjee
This is super simple program to test out the PineCone client.
You have to create an account in the PineCone website to get a free tier index.
This program shows how to load a model, store the elements in the vector store and then use similar searches using vectors
to find the nearest neighbors using cosine distance.
"""
class SimilarQuestions:
    def similar_questions_from_hugging_face(self, query:str):
        #Load dataset from huggign face
        dataset = load_dataset('quora', split='train[24000:32000]')
        #Print the first 10 questions
        print(dataset)
        qs = dataset['questions']
        questions =[]
        for question in qs:
            questions.extend(question['text'])

        questions = list(set(questions))
        #Now build embeddings
        from sentence_transformers import SentenceTransformer
        '''
        import torch
        device = torch.cuda.is_available()
        #Install CUDA and try again
        print (f'You are using {device}')
        '''
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(model)

        pc = PineCone(model)
        #First time uncomment this line
        #pc.update_or_insert(model, questions=questions)
        similar_questions = pc.find_similar(query, model)
        question =0
        for q in similar_questions:
            print(q)
            print(f'========== Question {question} ==========')
            question +=1
        return similar_questions
if __name__ == "__main__":
    ss = SimilarQuestions()
    ss.similar_questions_from_hugging_face('What is your name ?')





