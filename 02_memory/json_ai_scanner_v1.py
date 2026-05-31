from openai import OpenAI
import json
import pandas as pd
import os

client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))

is_active = True

messages=[
        {"role": "system", "content":"""You are an AI scanner and must always give result for any question in the json format, Return ONLY valid JSON.
            Do not include markdown.
            Do not include explanation text.:
            \"topic\": \"...\",
            \"difficulty\": \"...\",
            \"summary\": \"...\" """}
    ]

while is_active:
    user_input = input("Ask your question: (Say STOP to STOP) ")
    if user_input.lower() == 'stop':
        is_active = False
        continue

    messages.append(
        {"role":"user", "content":user_input}
    )
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages=messages
    )
    reply = response.choices[0].message.content
    data = json.loads(reply)
    formated_rep = f" Topic: {data['topic']}\n Difficulty: {data['difficulty']}\n Summary: {data['summary']}"
    print(formated_rep)
    messages.append(
        {"role":"assistant", "content": formated_rep}
    )
    with open("memory.json", "w") as file:
        json.dump(messages, file, indent=4)