from openai import OpenAI
import json
import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path='chroma_storage'
)

messages = []

# INGESTION

collection = chroma_client.get_or_create_collection(
    name= 'notes2_db'
)

count = collection.count()

if count > 0:
    print("Collection Exists!")
else:
    if os.path.exists('notes2.txt'):
        with open('notes2.txt', 'r') as file:
            content = file.read()
            pieces = content.split('\n\n')

        for i in range(len(pieces)):
            piece = pieces[i].strip()
            if piece == "":
                continue

            response = client.embeddings.create(
                model= 'text-embedding-3-small',
                input= piece
            )

            embedding = response.data[0].embedding
            collection.add(
                ids=[str(i)],
                documents=[piece],
                embeddings= [embedding],
                metadatas=[
                    {
                        "source":"notes2.txt",
                        "chunk_id": i
                    }
                ]
            )

# RETRIEVAL

is_true = True

if os.path.exists('memory_chunking.json'):
    with open('memory_chunking.json', 'r') as file:
        messages = json.load(file)

while is_true:
    query = input('STOP to STOP\nAsk a Question: ')
    if query.lower() == 'stop':
        break

    query_response = client.embeddings.create(
        model='text-embedding-3-small',
        input= query
    )
    query_embed = query_response.data[0].embedding

    results = collection.query(
        query_embeddings= [query_embed],
        n_results= 3
    )

    retrieved_text = results['documents'][0]
    good_chunks = []

    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    best_distance = results['distances'][0][0]

    for text, distance, metadata in zip(retrieved_text, distances, metadatas):
        print(f"Chunk Id: {metadata['chunk_id']}")
        print(f"Distance: {distance}")
        print(f"Chunk: {text}")

    for chunk, distance in zip(retrieved_text, distances):
        threshold = best_distance + 0.3
        if distance <= threshold:
            good_chunks.append(chunk)
    
    print("Best Distance: ", best_distance)
    print("\nNumber of good chunks: ", len(good_chunks))
    combined_text = "\n".join(good_chunks)

    if len(good_chunks) > 0:

        messages.append(
            {
                "role":"user",
                "content": query
            }
        )

        temp_messages= [
            {
                "role": "system",
                "content": f"""You are an AI assistant.
                            Answers the user's query using the retrieved context given below.
                            Retrieved Content: {combined_text}
                            If there is no available retrieved content then say you don't know the answer."""
            }
        ] + messages
        
        chat_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=temp_messages
        )

        chat_reply = chat_response.choices[0].message.content
        print(f'\nAnswer: \n{chat_reply}')

        messages.append(
            {
                "role":"assistant",
                "content":chat_reply
            }
        )

        if len(messages)>12:
            messages = messages[-12:]

        with open('memory_chunking.json','w') as file:
            json.dump(messages, file, indent=4)

        usage = chat_response.usage
        print(f'Prompt Token Usage: {usage.prompt_tokens}')
        print(f'Completions Token Usage: {usage.completion_tokens}')
        print(f'Total Token Usage: {usage.total_tokens}')
    else:
        print("I don't know (Not enough context provided.)")