from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = AsyncOpenAI(api_key= api_key)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

app = FastAPI()

@app.post('/ask', response_model=AnswerResponse)
async def ask(data: QuestionRequest):

    if data.question == "":
        raise HTTPException(
            status_code= 400,
            detail= "Question cannot be empty."
        )

    try:
        response =  await client.chat.completions.create(
            model='gpt-4.1-mini',
            messages= [
                {
                    "role":"system",
                    "content":"You are an AI assistant."
                },
                {
                    "role":"user",
                    "content": data.question
                }
            ]
        )
        reply= response.choices[0].message.content

        return {
            "answer": reply
        }
    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail= str(e)
        )