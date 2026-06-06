from fastapi import FastAPI
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question : str

app = FastAPI()

@app.get("/")
def home():
    return {
        "message":"Hello AI Systems"
    }

@app.get('/about')
def about():
    return {
        "name":"GenAI Learning API",
        "version":"1.0",
        "student":"suhana",
        "reporting to": "sensei-gpt"
    }

@app.post('/ask')
def ask(data:QuestionRequest):
    return {
        "recieved_question": data.question
    }