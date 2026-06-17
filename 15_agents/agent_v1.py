from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def search_docs(query):
    return f"Searching docs for: {query}"

def calculator(expression):
    return str(eval(expression))

tools = [
    {
        "type":"function",
        "function":{
            "name":"search_docs",
            "description":"Search AI and RAG knowledge base",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string"
                    }
                },
                "required":['query']
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name": "calculator",
            "description":"Perform mathematical calculations",
            "parameters":{
                "type":"object",
                "properties": {
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
            "role":"system",
            "content":"You are an AI Agent"
        },
        {
            "role":"user",
            "content":query
        }
    ],
    tools= tools
)

tool_calls = response.choices[0].message.tool_calls
if not tool_calls:
    print(response.choices[0].message.content)
    exit()

tool_call = tool_calls[0]

tool_name = tool_call.function.name
arguments= json.loads(tool_call.function.arguments)

if tool_name == 'search_docs':
    result = search_docs(arguments['query'])
elif tool_name == 'calculator':
    result = calculator(arguments['expression'])

messages = [
    {
        "role":"user",
        "content": query
    },
    response.choices[0].message,
    {
        "role":"tool",
        "tool_call_id": tool_call.id,
        "content":result
    }
]

final_response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages= messages
)

final_reply = final_response.choices[0].message.content
print(final_reply)

print(arguments)