from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import chromadb
import time

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path= 'chroma_storage'
)

# INGESTION

collection = chroma_client.get_or_create_collection(
    name='notes2_db'
)
count_collection = collection.count()

if count_collection > 0:
    print('Collection Exists in DB')
else:
    if os.path.exists('notes2.txt'):
        with open('notes2.txt', 'r') as file:
            content = file.read()
            pieces = content.split('\n\n')

        chunks = []
        for i in range(len(pieces)-1):
            overlap_text = pieces[i] + " " + pieces[i+1]
            chunks.append(overlap_text)

        for i, chunk in enumerate(chunks):
            chunk_response = client.embeddings.create(
                model='text-embedding-3-small',
                input=chunk
            )
            chunk_embed = chunk_response.data[0].embedding

            collection.add(
                ids=[str(i)],
                documents=[chunk],
                embeddings=[chunk_embed],
                metadatas=[
                    {
                        "source":"notes2.txt",
                        "chunk_id":i
                    }
                ]
            )

# RETRIEVAL
messages = []

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

MAX_RETRIES = 3
pass_number = 0
total_ques = 0

for item in eval_queries:
    total_ques +=1
    query = item['query']

    rewritten_reply = None

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt: {attempt+1}")
            rewritten_response = client.chat.completions.create(
                model= 'gpt-4.1-mini',
                messages=[
                    {
                        "role":"system",
                        "content": REWRITE_QUERY_V1
                    },
                    {
                        "role":"user",
                        "content":query
                    }
                ]
            )
            rewritten_reply = rewritten_response.choices[0].message.content
            print('Successfully Rewritten Query')
            break
        except Exception as e:
            print(f"Attempt failed: {e}")
            if attempt < MAX_RETRIES -1:
                time.sleep(attempt + 1)

    if rewritten_reply ==  None:
        print("Retry attmept failed")
        print("Using original query.")
        rewritten_reply = query

    try:
        expected_chunk = item['expected'].lower()

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
        print(retrieved_text)

        ACCURACY_LEVEL = 3
        exists = False

        for chunk in retrieved_text[:ACCURACY_LEVEL]:
            chunk = chunk.lower()
            if expected_chunk in chunk:
                print(f"Chunk: {chunk}")
                print(f"Expected: {expected_chunk}")
                exists = True
                break
            
        if exists:
            pass_number +=1
            print(f'{query} : PASS')
        else:
            print(f'{query} : FAIL')

    except Exception as e:
        print(str(e))

print(f"Score: {pass_number}/{total_ques}")
print(f"Accuracy: {(pass_number/total_ques)*100}")