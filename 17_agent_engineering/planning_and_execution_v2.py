from openai  import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from math_agent import math_agent
from general_agent import general_agent

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def rag_agent(step):
    return f'Searching doc for {step}'

class Plan(BaseModel):
    steps : list[str]

class RouterResponse(BaseModel):
    routes: list[str]

# PLANNING AGENT

query = input("Ask a Question: ")

planner_response = client.beta.chat.completions.parse(
    model = 'gpt-4.1-mini',
    messages=[
        {
            'role':'system',
            'content':"""Break the user's request
                        into a list of executable steps.

                        Return only the plan."""
        },
        {
            'role':'user',
            'content': query
        }
    ],
    response_format= Plan
)

planner_reply = planner_response.choices[0].message.parsed

plans = planner_reply.steps

# ROUTER AGENT

results = []
for step in plans:
    print(f"Steps: {step}")
    router_response = client.beta.chat.completions.parse(
        model='gpt-4.1-mini',
        messages=[
            {
                'role':'system',
                'content':  """
                                You are a routing agent.

                                Available Agents:

                                rag:
                                Use ONLY when the answer is expected
                                to come from the AI/RAG knowledge base
                                containing:
                                - embeddings
                                - vector databases
                                - ChromaDB
                                - FastAPI
                                - AI concepts

                                math:
                                Use for calculations.

                                general:
                                Use for general world knowledge,
                                explanations, writing,
                                science questions,
                                and conversation.

                                Return ONLY:
                                rag
                                math
                                general
                            """
            },
            {
                'role':'user',
                'content':step
            }
        ],
        response_format= RouterResponse
    )
    routes = router_response.choices[0].message.parsed.routes
    print(f"Routes: {routes}")
    for route in routes:
        if route == 'rag':
            results.append(rag_agent(step))
        elif route == 'math':
            results.append(math_agent(client, step))
        else:
            results.append(general_agent(client, step))

print('Answer:')
for result in results:
    print(result)