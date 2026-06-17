from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

def search_docs(query):
    return f"Searching doc for {query}"

def calculator(expression):
    return str(eval(expression))

tools = [
    {
        "type":"function",
        "function":{
            "name":"search_docs",
            "decription":"Search AI and RAG knowledge base",
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
            "name":"calculator",
            "decription":"Perform mathematical calculations",
            "parameters":{
                "type":"object",
                "properties":{
                    "expression":{
                        "type":"string"
                    }
                },
                "required":['expression']
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
            "content":query
        }
    ],
    tools = tools
)

tool_calls = response.choices[0].message.tool_calls

if not tool_calls:
    print(response.choices[0].message.content)
    exit()

messages = [
        {
            "role":"user",
            "content": query
        },
        response.choices[0].message
    ]

for tool_call in tool_calls:
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if tool_name == 'search_docs':
        results = search_docs(arguments['query'])
    elif tool_name == 'calculator':
        results = calculator(arguments['expression'])

    messages.append(
        {
            "role":"tool",
            "tool_call_id": tool_call.id,
            "content": results
        }
    )

final_response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages= messages
)

final_reply = final_response.choices[0].message.content
print(f"Answer: {final_reply}")
print(f"Numebr of tool calls: {len(tool_calls)}")