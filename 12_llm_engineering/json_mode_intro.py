from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model = 'gpt-4.1-mini',
    messages=[
        {
            "role":"system",
            "content":"You must return a valid JSON"
        },
        {
            "role":"user",
            "content":"Explain RAG with topic, difficulty and summary."
        }
    ],
    response_format= {
        "type":"json_object"
    }
)

reply = response.choices[0].message.content
# print(reply)
# print(type(reply))

data = json.loads(
    reply
)

print(data['topic'])