import json

def math_agent(client, query):

    def calculator(expression):
        return str(expression)
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

    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {
                'role':'user',
                'content':query
            }
        ],
        tools = tools
    )
    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls:
        return response.choices[0].message.content

    messages = [
        {
            'role':'user',
            'content': query
        },
        response.choices[0].message
    ]

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

    return final_reply