from openai import OpenAI
from dotenv import load_dotenv
from retrieval import retrieve_context
import os
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key= api_key)

def calculator(expression):
    return str(eval(expression))

tools = [
    {
        "type":"function",
        "function":{
            "name":"retrieve_context",
            "description": "Search AI and RAG knowledge base",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string"
                    }
                },
                "required":["query"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculator",
            "description":"Perform mathematical calculations",
            "parameters":{
                "type":"object",
                "properties":{
                    "expression":{
                        "type":"string"
                    }
                },
                "required":["expression"]
            }
        }
    }
]

query = input("Ask a Question: ")

response = client.chat.completions.create(
    model = 'gpt-4.1-mini',
    messages=[
        {
            "role":"user",
            "content": query
        }
    ],
    tools = tools
)

tool_calls = response.choices[0].message.tool_calls

if not tool_calls:
    print(response.choices[0].message.content)
    exit()

assistant_message = response.choices[0].message

messages = [
    {
        "role": "user",
        "content": query
    },
    assistant_message
]

for tool_call in tool_calls:
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if tool_name == 'retrieve_context':
        result = retrieve_context(client, arguments['query'])
    elif tool_name == 'calculator':
        result = calculator(arguments['expression'])

    print(f"Tools Name: {tool_name}")

    messages.append(
        {
            "role":"tool",
            "tool_call_id":tool_call.id,
            "content": result
        }
    )

final_response = client.chat.completions.create(
    model = 'gpt-4.1-mini',
    messages= messages
)

final_reply = final_response.choices[0].message.content
print(f"Answer: {final_reply}")