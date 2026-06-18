from openai  import OpenAI
from dotenv import load_dotenv
from math_agent import math_agent
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def rag_agent(query):
    return f"Searching doc for {query}"

def general_agent(query):
    return f"Answer for {query}"

query = input('Ask a Question: ')
router_response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role":"system",
            "content":"""
            You are a routing agent.

            Available Agents:

            rag:
            Use for AI, RAG, embeddings,
            vector databases, FastAPI,
            and knowledge-base questions.

            math:
            Use for arithmetic and calculations.

            general:
            Use for greetings, jokes,
            writing, and general conversation.

            Return ONLY:
            rag
            math
            general
            """
        },
        {
            "role":"user",
            "content": query
        }
    ]
)

route = router_response.choices[0].message.content
print(f"Route: {route}")

if route == 'rag':
    result= rag_agent(query)
elif route == 'math':
    result= math_agent(query)
else:
    result= general_agent(query)

print(f"Answer: {result}")