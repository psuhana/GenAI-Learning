from openai import OpenAI
from fastapi import FastAPI

## THE STREAMING WRAPPER
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

class QuestionRequest(BaseModel):
    question : str

def generate(response):
    for chunk in response:
        token = chunk.choices[0].delta.content

        if token:
            print(token, end="")
            yield token

app = FastAPI()

@app.post('/stream')
def stream(data: QuestionRequest):
    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {
                "role":"system",
                "content":"You are an AI Assistant who specializes with AI/RAG queries and answers in detailed."
            },
            {
                "role":"user",
                "content": data.question
            }
        ],
        stream= True
    )

    return StreamingResponse(
        generate(response),
        media_type='text/plain'
    )