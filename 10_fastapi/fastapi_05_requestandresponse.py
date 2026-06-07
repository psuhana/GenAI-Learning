from fastapi import FastAPI
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

app = FastAPI()

@app.post('/ask', response_model=AnswerResponse)
def ask(data: QuestionRequest):
    return {
        "answer": data.question
    }