from openai import OpenAI
import os

client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

messages= [
        {"role": "system", "content": "You are an AI assistant"}
    ]

is_active = True

while is_active:
    user_input= input("Ask your question (Stop to STOP agent): ")
    if user_input.lower() == 'stop':
        is_active = False
        continue

    messages.append(
        {"role": "user", "content": user_input}
    )

    response = client.chat.completions.create(
        model= "gpt-4.1-mini",
        messages= messages
    )
    reply = response.choices[0].message.content
    print(messages)
    print(reply)
    messages.append(
        {"role":"assistant", "content": reply}
    )