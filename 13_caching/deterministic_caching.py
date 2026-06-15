from openai import OpenAI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
import json
import chromadb

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key= api_key)
chroma_client = chromadb.PersistentClient(
    path= r"C:\Users\suhan\Desktop\LLM\chroma_storage"
)

class QuestionRequest(BaseModel):
    question : str

def reject():
    raise HTTPException(
        status_code= 400,
        detail='There is an issue with the query.'
    )

cache = {}

if os.path.exits('cache.json'):
    with open('cache.json', 'r') as file:
        cache = json.load(file)

collection = chroma_client.get_or_create_collection(
    name='notes2_db'
)
count_collection = collection.count()
collection_exist = False

if count_collection > 0:
    collection_exist = True
else:
    if os.path.exists('notes2.txt'):
        with open('notes2.txt', 'r') as file:
            content = file.read()
            pieces = content.split('\n\n')

        chunks = []
        for i in range(len(pieces) - 1):
            overlap_text = pieces[i] + " " + pieces[i+1]
            chunks.append(overlap_text)

        for i, chunk in enumerate(chunks):
            chunk_response = client.embeddings.create(
                model='text-embedding-3-small',
                input= chunk
            )
            chunk_embed = chunk_response.data[0].embedding
            collection.add(
                ids=[str(i)],
                documents=[chunk],
                embeddings=[chunk_embed],
                metadatas=[
                    {
                        "source":"notes2.txt",
                        "chunk_id": i
                    }
                ]
            )

# RETRIEVAL

## PROMPTS
REWRITE_QUERY_VERSION = 'v1'
REWRITE_QUERY = """ You are an AI Rewiter for AI/RAG Assistance.
                    Rewrite user queries into short retrieval friendly search queries.
                    Rules:
                    - Keep queries concise.
                    - Preserve conversational meaning.
                    - Preserve AI/RAG domain context.
                    - DO NOT answer the question
                    - DO NOT explain concepts
                    - Return only the rewritten query.
                    """

app = FastAPI()

@app.post('/ask')
def ask(data:QuestionRequest):
    # Guardrails
    query = data.question.lower().strip()
    if query == "":
        reject()
    if len(query) > 50000:
        reject()
    
    # DETERMINISTIC CHECK
    if query in cache:
        print("CACHE HIT ⚡")
        return cache[query]
    else:
        print("CACHE MISS ❌")
        rewritten_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[
                {
                    "role":"system",
                    "content": REWRITE_QUERY
                },
                {
                    "role":"user",
                    "content":query
                }
            ],
        )
        rewritten_reply = rewritten_response.choices[0].message.content
        query_response = client.embeddings.create(
            model='text-embedding-3-small',
            input= rewritten_reply
        )
        query_embed = query_response.data[0].embedding
        results = collection.query(
            query_embeddings=[query_embed],
            n_results= 10
        )

        retrieved_text = results['documents'][0]
        top_chunks = retrieved_text[:2]

        combined_text = '\n'.join(top_chunks)

        chat_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[
                {
                    "role":"system",
                    "content":f"""You are AI Assistant.
                                    Answer user's queries using the context of the retreived content below.
                                    Retrieved Text: {combined_text}
                                    Answer ONLY using retrieved context.
                                    If answer cannot be found in retrieved context,
                                    reply exactly:
                                    "I don't know."""
                },
                {
                    "role":"user",
                    "content": query
                }
            ]
        )
        chat_reply = chat_response.choices[0].message.content

        cache[query] = chat_reply

        log_data = [
            {
                "collection_count": count_collection,
                "query": query,
                "rewritten query": rewritten_reply,
                "rewritten query version": REWRITE_QUERY_VERSION,
                "retrieved chunks": top_chunks,
                "answer": chat_reply
            }
        ]

        print(json.dumps(log_data, indent=4))

        with open('cache.json', 'w') as file:
            json.dump(cache, file, indent=4)

        with open('logs.json', 'a') as file:
            json.dump(log_data, file, indent=4)
        return chat_reply    