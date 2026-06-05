from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages= [
        {
            "role":"system",
            "content": """You are an AI assistant who answers questions in the AI/RAG domain context."""
        },
        {
            "role":"user",
            "content": "What is RAG?"
        }
    ],
    stream= True
)

for chunk in response:
    token = chunk.choices[0].delta.content

    if token:
        print(token, end="")