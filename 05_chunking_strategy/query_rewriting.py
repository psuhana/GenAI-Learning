from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import chromadb

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key= api_key)
chroma_client = chromadb.PersistentClient(
    path = 'chroma_storage'
)

messages = []
is_true = True

# INGESTION

collection = chroma_client.get_or_create_collection(
    name='notes2_db'
)

count = collection.count()

if count == 0:
    if os.path.exists('notes2.txt'):
        with open('notes2.txt', 'r') as file:
            content= file.read()
            pieces = content.split('\n\n')

        chunks = []
        for i in range(len(pieces) - 1):
            overlap_text = pieces[i] + " " + pieces[i+1]
            chunks.append(overlap_text)

        for i, chunk in enumerate(chunks):
            
            chunk_response = client.embeddings.create(
                model='text-embedding-3-small',
                input = chunk
            )

            chunk_embed = chunk_response.data[0].embedding

            collection.add(
                ids = [str(i)],
                documents=[chunk],
                embeddings=[chunk_embed],
                metadatas=[
                    {
                        "source":"notes2.txt",
                        "chunk_id": i
                    }
                ]
            )

else:
    print('Collection Exists in DB.')

# RETRIEVAL

if os.path.exists('memory_chunking.json'):
    with open('memory_chunking.json', 'r') as file:
        messages = json.load(file)

while is_true:
    query = input('Stop for STOP\nAsk a Question: ')
    if query.lower() == 'stop':
        break

    refined_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages= [
                {
                    "role":"system",
                    "content": """You are a query rewriter for an AI and RAG assistant.
                                Rewrite user queries into SHORT retrieval-friendly search queries.
                                Rules:
                                - Keep queries concise
                                - Preserve conversational meaning
                                - Preserve AI/RAG domain context
                                - Do NOT answer the question
                                - Do NOT explain concepts
                                - Return ONLY the rewritten query
                                """
                }
            ]
            + messages[-4:]+
            [
                {
                "role":"user",
                "content":query
                }
            ]
    )

    refined_query = refined_response.choices[0].message.content
    print("Refined Query: ", refined_query)

    query_response = client.embeddings.create(
        model='text-embedding-3-small',
        input= refined_query
    )

    query_embed = query_response.data[0].embedding

    results = collection.query(
        query_embeddings= [query_embed],
        n_results= 3
    )

    retrieved_text = results['documents'][0]
    distances = results['distances'][0]

    good_chunks = []
    best_distance = results['distances'][0][0]

    for chunk, distance in zip(retrieved_text, distances):
        threshold = best_distance + 0.3
        if distance <= threshold:
            good_chunks.append(chunk)

    combined_chunk = "\n".join(good_chunks)
    retained_chunks = len(good_chunks)

    print('Number of good chunks: ', retained_chunks)

    if retained_chunks > 0:

        messages.append(
            {
                "role":"user",
                "content": query
            }
        )

        temp_message = [{
            "role":"system",
            "content": f"""You are AI Assistant.
                            Answer user's queries only using the context given in the retrieved content below.
                            Retrieved Content: {combined_chunk}
                            If answer to query not in the retained context, reply I don't know."""
        }] + messages

        chat_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages= temp_message
        )

        chat_reply = chat_response.choices[0].message.content
        print(f"Answer:\n {chat_reply}")

        messages.append(
            {
                "role":"assistant",
                "content":chat_reply
            }
        )

        if len(messages)>12:
            messages = messages[-12:]

        with open('memory_chunking.json', 'w') as file:
            json.dump(messages, file, indent=4)

        usage = chat_response.usage
        print('Total token: ', usage.total_tokens)

    else:
        print('Query out of context.')