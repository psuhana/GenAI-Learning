from openai  import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def calculator(expression):
    return str(eval(expression))

tools = [
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
                "required":['expression']
            }  
        }
    }
]

messages = []

messages.append(
    {
        'role':'system',
        'content':'You are a helpful AI assistant.'
    }
)

active = True

while active:
    query = input("Ask a Question: ")
    if query.lower() == 'stop':
        active = False
        break

    messages.append(
        {
            'role':'user',
            'content':query
        }
    )

    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools = tools
    )
    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls:
        reply = response.choices[0].message.content
        print(reply)

        messages.append(
            {
                'role':'assistant',
                'content': reply
            }
        )
        continue

    messages.append(
       response.choices[0].message
    )

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if tool_name == 'calculator':
            result = calculator(arguments['expression'])
        
        messages.append(
            {
                'role':'tool',
                'tool_call_id':tool_call.id,
                'content': result
            }
        )

    final_response = client.chat.completions.create(
        model = 'gpt-4.1-mini',
        messages= messages
    )
    final_reply = final_response.choices[0].message.content

    messages.append(
        {
            'role':'assistant',
            'content':final_reply
        }
    )

    if len(messages)> 12:
        messages = [messages[0] + messages[-11:]]