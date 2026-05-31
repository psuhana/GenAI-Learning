from openai import OpenAI
import os
import numpy as np
import json

api_key = os.getenv('OPENAI_API_KEY')

if api_key == None:
    print('No key found')
else:
    client = OpenAI(api_key=api_key)

messages = []
is_true = True

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    mag = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    sim = dot/mag
    return sim

if os.path.exists('notes2.txt'):
    with open('notes2.txt', 'r') as file:
        content = file.read()
        pieces = content.split('.')

        embeddings = []

        for piece in pieces:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input= piece
            )
            embed = response.data[0].embedding
            embeddings.append(embed)
else:
    print("Notes does not exist!")
    is_true = False

while is_true:
    quest = input('To stop say stop\nAsk your question: ')
    if quest.lower() == 'stop':
        break
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input = quest
    )
    embed_user = response.data[0].embedding

    use_embed = []
    best_sim = 0
    
    for i in range(len(embeddings)):
        check = cosine_similarity(embeddings[i], embed_user)
        if check > best_sim:
            best_sim = check
            best_chunk = i
        
    use_embed = pieces[best_chunk]

    messages.append(
        {"role":"system", 
         "content": f"You are an AI assistant who uses {use_embed} to answer user questions."}
    )

    messages.append(
        {"role": "user", "content": quest}
    )

    response = client.chat.completions.create(
        model= "gpt-4.1-mini",
        messages= messages     
    )
    reply = response.choices[0].message.content
    print(reply)
    messages.append(
        {
            "role":"assistant",
            "content":reply
        }
    )
    with open('memory_RAG.json', 'w') as file:
        json.dump(messages, file, indent=4)