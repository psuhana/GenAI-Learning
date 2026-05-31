from openai import OpenAI
import os
import json

client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))

isActive = True

if os.path.exists('memory.json'):
    with open('memory.json','r') as file:
        messages = json.load(file)
else:
    messages=[
            {"role": "system", "content":"""You are an AI scanner and must always give result for any question in the json format, Return ONLY valid JSON.
                Do not include markdown.
                Do not include explanation text.:
                \"topic\": \"...\",
                \"difficulty\": \"...\",
                \"summary\": \"...\" """}
        ]

while(isActive):
    userInput = input("Enter your question: STOP to stop ")
    if userInput.lower() == 'stop':
        isActive = False
        continue

    messages.append(
        {'role':'user', 'content':userInput}
    )

    response = client.chat.completions.create(
        model = 'gpt-4.1-mini',
        messages= messages
    )

    reply = response.choices[0].message.content
    print(reply)
    messages.append(
        {'role':'assistant', 'content':reply}
    )

    with open('memory.json', 'w') as file:
        json.dump(messages, file, indent=4)