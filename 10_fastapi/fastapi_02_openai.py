from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key= api_key)

class QuestionRequest(BaseModel):
    question : str

app = FastAPI()

@app.post('/ask')
def ask(data:QuestionRequest):
    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {
                "role":"system",
                "content":"You are an AI assistant specialized in AI systems and RAG."
            },
            {
                "role":"user",
                "content": data.question
            }
        ]
    )

    answer = response.choices[0].message.content

    return{
        "answer": answer
    }

@app.get('/health')
def health():
    return {
        "status":"healthy"
    }

@app.get('/version')
def version():
    return {
        "api_version": "1.0",
        "llm_model":"gpt-4.1-mini"
    }