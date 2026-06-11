from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def multiply(a,b):
    return a * b

response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages=[
        {
            "role":"user",
            "content":"Can you calculate 15 times 7?"
        }
    ],
    tools = [
        {
            "type":"function",
            "function": {
                "name":"multiply",
                "description":"Multiply two numbers.",
                "parameters":{
                    "type":"object",
                    "properties": {
                        "a": {"type":"number"},
                        "b": {"type":"number"},
                    },
                    "required": ['a', 'b']
                }
            }
        }
    ]
)

tool_call = response.choices[0].message.tool_calls[0]

name = tool_call.function.name
print(type(name))
arguments = json.loads(tool_call.function.arguments)
a = arguments['a']
b = arguments['b']

result = multiply(a,b)
print(result)