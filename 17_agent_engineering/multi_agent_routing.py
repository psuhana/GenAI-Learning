from openai  import OpenAI
from dotenv import load_dotenv
from math_agent import math_agent
from pydantic import BaseModel
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def rag_agent(query):
    return f"Searching doc for {query}"

def general_agent(query):
    return f"Answer for {query}"

class RouterResponse(BaseModel):
    routes: list[str]

query = input('Ask a Question: ')
router_response = client.beta.chat.completions.parse(
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

                    Return a list of routes.

                    Examples:

                    Question:
                    What is ChromaDB?
                    Output:
                    ["rag"]

                    Question:
                    What is 25 * 18?
                    Output:
                    ["math"]

                    Question:
                    What is ChromaDB and what is 25 * 18?
                    Output:
                    ["rag","math"]
                    """
        },
        {
            "role":"user",
            "content": query
        }
    ],
    response_format= RouterResponse
)

parsed = router_response.choices[0].message.parsed
routes = parsed.routes
print(f"Route: {routes}")

results = []
for route in routes:
    if route == 'rag':
        results.append(rag_agent(query))
    elif route == 'math':
        results.append(math_agent(client,query))
    else:
        results.append(general_agent(query))

for result in results:
    print(f"Answer: {result}")