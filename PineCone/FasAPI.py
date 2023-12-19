

from fastapi import FastAPI
from SemanticSearchWithPineCone import SimilarQuestions
#uvicorn FastAPI:app --reload


app = FastAPI()
#uvicorn main:app --reload
@app.get("/similar_questions/{token}")
def read_root(token):
    ss = SimilarQuestions()
    return {"Similar Questions  is ": ss.similar_questions_from_hugging_face(token)}
