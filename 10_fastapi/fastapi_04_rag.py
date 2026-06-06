from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import os
import chromadb

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path = r'C:\Users\suhan\Desktop\LLM\chroma_storage'
)

class QuestionRequest(BaseModel):
    question : str

messages = []

# INGESTION

collection = chroma_client.get_or_create_collection(
    name = 'notes2_db'
)

count_chroma = collection.count()
print("Collection Count:", count_chroma)

if count_chroma > 0:
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
                input= chunk
            )
            chunk_embed = chunk_response.data[0].embedding

            collection.add(
                ids= [str(i)],
                documents= [chunk],
                embeddings= [chunk_embed],
                metadatas=[
                    {
                        "source":"notes2.txt",
                        "chunk_id": i
                    }
                ]
            )
    
    print ('Content added to DB.')

# RETRIEVAL

app = FastAPI()

@app.get('/health')
def health():
    return {
        "status":"healthy"
    }

@app.get('/version')
def version():
    return {
        "api_version":"1.1",
        "llm_model":"gpt-4.1-mini"
    }

@app.post('/rag')
def rag_pipeline(data: QuestionRequest):
    query = data.question

    rewritten_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages= [
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
        ]
    )
    rewritten_reply = rewritten_response.choices[0].message.content
    print(f'Rewritten Query: {rewritten_reply}')

    query_response = client.embeddings.create(
        model='text-embedding-3-small',
        input= rewritten_reply
    )
    query_embed = query_response.data[0].embedding

    results = collection.query(
        query_embeddings= [query_embed],
        n_results= 10
    )

    retrieved_text = results['documents'][0]
    print(f'Retrieved Text: {retrieved_text}')
    distances = results['distances'][0]

    query_words = rewritten_reply.lower().split()

    scored_chunks = []

    for chunk, distance in zip(retrieved_text, distances):
        keyword_score = 0
        chunk_words = chunk.lower().split()

        for word in query_words:
            if word in chunk_words:
                keyword_score += 1

        final_score = keyword_score - distance
        scored_chunks.append(
            (final_score, chunk, distance, keyword_score)
        )

    scored_chunks = sorted(scored_chunks, reverse=True)

    final_chunks = scored_chunks[:2]

    top_chunks = []
    for final_score, chunk, distance, keyword_score in final_chunks:
        print(
            f"Score={final_score} "
            f"Distance={distance} "
            f"Keywords={keyword_score}"
        )
        print(f"Chunk: {chunk}")
        top_chunks.append(chunk)

    combined_text = "\n".join(top_chunks)
    print("\nRetrieved Context:")
    print(combined_text)

    chat_response = client.chat.completions.create(
        model= "gpt-4.1-mini",
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
                "content": rewritten_reply
            }
        ]
    )
    chat_reply = chat_response.choices[0].message.content
    return {
        "answer": chat_reply
    }