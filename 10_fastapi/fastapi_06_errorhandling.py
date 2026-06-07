from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key= api_key)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

app = FastAPI()

@app.post('/ask', response_model= AnswerResponse)
def ask(data: QuestionRequest):

    if data.question == "":
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

    try:
        response = client.chat.completions.create(
            model='gpt-potato-9000',
            messages= [
                {
                    "role":"system",
                    "content": "You are an AI Assistant specialized in AI and RAG systems."
                },
                {
                    "role":"user",
                    "content": data.question
                }
            ],
            stream= True
        )
        
        reply = response.choices[0].message.content
        return {
            "answer": reply
        }
    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail= str(e)
        )


    # except Exception as e:
    #   return {
    #        "answer": f"Error: {str(e)}"
    #   }