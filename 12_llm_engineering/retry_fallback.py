from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import chromadb
import time

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path = r"C:\Users\suhan\Desktop\LLM\chroma_storage"
)

def reject():
    raise HTTPException(
        status_code= 400,
        detail= 'There was an issue with the query!'
    )

class QuestionRequest(BaseModel):
    question: str

class ClassifierResponse(BaseModel):
    query_ok : bool

messages = []

app = FastAPI()

# INGESTION

collection = chroma_client.get_or_create_collection(
    name="notes2_db"
)

count_collection = collection.count()

if count_collection > 0:
    print('Collection exists in DB.')
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
                input= chunk
            )
            chunk_embed = chunk_response.data[0].embedding

            collection.add(
                ids = [str(i)],
                documents=[chunk],
                embeddings=[chunk_embed],
                metadatas=[
                    {
                        "source":"notes2.txt",
                        "chunk_id":i
                    }
                ]
            )
        
    print('Collection added to DB')

# RETRIEVAL

banned_words = [
    'malware',
    'hack',
    'virus'
]

@app.post('/ask')
def ask(data:QuestionRequest):
    query = data.question

    if query.strip() == "":
        reject()

    if len(query)>100:
        reject()

    for word in banned_words:
        if word in query.lower():
            reject()

    if os.path.exists('memory_chunking.json'):
        with open('memory_chunking.json', 'r') as file:
            messages = json.load(file)

    MAX_RETRIES = 3
    rewritten_reply = None

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt: {attempt+1}")
            rewritten_response = client.chat.completions.create(
                model='gpt-potato-9000', # INTENTIONAL FAIL
                messages=[
                    {
                        "role":"system",
                        "content": """ You are an AI Rewiter for AI/RAG Assistance.
                            Rewrite user queries into short retrieval friendly search queries.
                            Rules:
                            - Keep queries concise.
                            - Preserve conversational meaning.
                            - Preserve AI/RAG domain context.
                            - DO NOT answer the question
                            - DO NOT explain concepts
                            - Return only the rewritten query.
                            """
                    },
                    {
                        "role":"user",
                        "content": query
                    }
                ],
            )
            rewritten_reply = rewritten_response.choices[0].message.content
            print("Rewrite Successful!")
            print(f"Rewritten Reply: {rewritten_reply}")
            break

        except Exception as e:
            print(f'Attempt error: {e}')
            if attempt < MAX_RETRIES -1:
                time.sleep(attempt + 1) # LINEAR BACKOFF

    if rewritten_reply == None:
        print('Rewrite attempt failed')
        print('Using original query.')
        rewritten_reply = query

    try:
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
        # distances = results['distances'][0]
        # query_words = rewritten_reply.lower().split()

        # RERANKING        
        # select_chunks = []
        # for chunk, distance in zip(retrieved_text, distances):
        #    keyword_score = 0
        #    chunk_words = chunk.lower().split()

        #    for word in query_words:
        #        if word in chunk_words:
        #            keyword_score +=1

        #    final_score = keyword_score - distance
        #    select_chunks.append(
        #        (final_score, chunk)
        #    )

        # select_chunks = sorted(select_chunks, reverse=True)
        # select_chunks = select_chunks[:2]
        #for final_score, chunk in select_chunks:
        #    print(f"Final Score: {final_score} | ")
        #    print(f"Chunk: {chunk}")
        #    top_chunks.append(chunk)

        top_chunks = retrieved_text[:2]
        combined_text = "\n".join(top_chunks)

        messages.append(
            {
                "role":"user",
                "content": query
            }
        )

        temp_message = [
            {
                "role":"system",
                "content":f"""You are AI Assistant.
                    Answer user's queries using the context of the retreived content below.
                    Retrieved Text: {combined_text}
                    Answer ONLY using retrieved context.
                    If answer cannot be found in retrieved context,
                    reply exactly:
                    "I don't know."""
            }
        ] + messages

        chat_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages= temp_message
        )
        chat_reply = chat_response.choices[0].message.content

        messages.append(
            {
                "role":"assistant",
                "content": chat_reply
            }
        )

        if len(messages) > 12:
            messages = messages[-12:]

        with open('memory_chunking.json', 'w') as file:
            json.dump(messages, file, indent = 4)

        return chat_reply

    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail=  str(e)
        )