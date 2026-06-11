from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key= api_key)

class TopicResponse(BaseModel):
    topic: str
    difficulty: str
    summary: str

response = client.beta.chat.completions.parse(
    model='gpt-4.1-mini',
    messages=[
        {
            "role":"user",
            "content":"Explain RAG"
        }
    ],
    response_format=TopicResponse
)

result = response.choices[0].message.parsed

print(f"Topic: {result.topic}")
print(f"Difficulty: {result.difficulty}")
print(f"Summary: {result.summary}")