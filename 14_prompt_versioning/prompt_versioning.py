from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingestion import ingestion
import os
import chromadb
import redis
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path = r"C:\Users\suhan\Desktop\LLM\chroma_storage"
)
redis_client = redis.Redis(
    host = 'localhost',
    port = 6379,
    decode_responses= True
)

knowledge_collection = chroma_client.get_or_create_collection(
    name='notes2_db'
)
count_collection = knowledge_collection.count()

if count_collection == 0:
    ingestion(client, knowledge_collection)
else:
    print('Collection present in DB.')

# RETRIEVAL

def reject():
    raise HTTPException(
        status_code = 400,
        details = 'There is an issue in the query'
    )

REWRITE_QUERY_V1 = """ You are an AI Rewiter for AI/RAG Assistance.
                    Rewrite user queries into short retrieval friendly search queries.
                    Rules:
                    - Keep queries concise.
                    - Preserve conversational meaning.
                    - Preserve AI/RAG domain context.
                    - DO NOT answer the question
                    - DO NOT explain concepts
                    - Return only the rewritten query.
                    """

REWRITE_QUERY_V2 = """
                    You are an AI Query Rewriter.

                    Rewrite the query while preserving important AI/RAG terminology.

                    Rules:
                    - Keep technical keywords.
                    - Expand abbreviations if useful.
                    - Return only the rewritten query.
                    """

ACTIVE_REWRITE_VERSION = 'v2'

eval_queries = [
    {
        "query":"What is an embedding?",
        "expected":"Embeddings convert text into numerical vector representations"
    },
    {
        "query":"What is ChromaDB?",
        "expected":"ChromaDB is a vector database"
    },
    {
        "query":"What is FastAPI?",
        "expected":"FastAPI is commonly used"
    }
]

PASS = 0
question_num = 0
eval_data = []

for item in eval_queries:
    question_num += 1
    query = item['query']
    expected = item['expected'].lower().strip()

    if query == '':
        reject()
    if len(query) > 5000:
        reject()
    
    cached_answer = redis_client.get(query)
    if cached_answer:
        print('CACHE HIT!')
        print(cached_answer)
    else:
        rewritten_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[
                {
                    'role':'system',
                    'content': REWRITE_QUERY_V2
                },
                {
                    'role':'user',
                    'content': query
                }
            ]
        )
        rewritten_reply = rewritten_response.choices[0].message.content
        query_response = client.embeddings.create(
            model='text-embedding-3-small',
            input= rewritten_reply
        )
        query_embed = query_response.data[0].embedding

        results = knowledge_collection.query(
            query_embeddings=[query_embed],
            n_results=10
        )

        retrieved_text = results['documents'][0]
        hit = False
        for chunk in retrieved_text:
            if expected in chunk.lower().strip():
                hit = True
                PASS += 1
                break
        
        if hit:
            print(f"{query}: PASS ✅")
        else:
            print(f"{query}: FAIL 😭")

        log_data = {
            "Collection Count": count_collection,
            "Query":query,
            "Rewritten Query": rewritten_reply,
            "Active Rewrite Prompt Vesion": ACTIVE_REWRITE_VERSION,
            "Hit": hit
        }
        print(json.dumps(log_data, indent = 4))
        eval_data.append(log_data)

print(f"Passes: {PASS}")
print(f"Number of Question: {question_num}")
accuracy = (PASS/question_num)*100
print(f"Accuracy: {accuracy}")