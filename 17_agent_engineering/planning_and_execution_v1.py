from openai  import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def calculator(expression):
    return str(eval(expression))

class Plan(BaseModel):
    steps : list[str]

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

# EXECUTION AGENT

execution_results = []
for i, step in enumerate(plans):
    print(f"Step {i+1}. {step}")
    execution_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {
                'role':'system',
                'content': f"""
                            Execute this step:
                            {step} """
            }
        ]
    )
    execution_reply = execution_response.choices[0].message.content
    execution_results.append(execution_reply)

for ans in execution_results:
    print(ans)