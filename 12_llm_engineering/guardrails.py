from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import chromadb

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

    try:
        guardrail_response = client.chat.completions.parse(
            model='gpt-4.1-mini',
            messages=[
                {
                    "role":"system",
                    "content":"""
                                    You are a Query Classifier for an AI and RAG Assistant.

                                    Your task is to determine whether a user query should be allowed.

                                    A query is allowed ONLY IF:
                                    1. It is related to AI, Machine Learning, LLMs, RAG, Vector Databases, Agents, 
                                    Prompt Engineering, FastAPI, Python for AI, or similar AI Engineering topics.
                                    2. It does not contain harmful, malicious, illegal, dangerous, or abusive requests.

                                    Examples:

                                    User: "What is Retrieval-Augmented Generation?"
                                    Result: {"query_ok": true}

                                    User: "Explain ChromaDB."
                                    Result: {"query_ok": true}

                                    User: "How do I build an AI chatbot?"
                                    Result: {"query_ok": true}

                                    User: "How do I create malware?"
                                    Result: {"query_ok": false}

                                    User: "How do I hack a website?"
                                    Result: {"query_ok": false}

                                    User: "What is the capital of France?"
                                    Result: {"query_ok": false}

                                    User: "Write a chocolate cake recipe."
                                    Result: {"query_ok": false}

                                    Return only the structured output.
                                    """
                },
                {
                    "role":"user",
                    "content": query
                }
            ],
            response_format= ClassifierResponse
        )
        guardrail_reply = guardrail_response.choices[0].message.parsed

        if guardrail_reply.query_ok:
            rewritten_response = client.chat.completions.create(
                model='gpt-4.1-mini',
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

            print(f"Rewritten Reply: {rewritten_reply}")

            query_response = client.embeddings.create(
                model='text-embedding-3-small',
                input= rewritten_reply
            )
            query_embed = query_response.data[0].embedding

            results = collection.query(
                query_embeddings=[query_embed],
                n_results= 10
            )

            query_words = rewritten_reply.lower().split()

            retrieved_text = results['documents'][0]
            distances = results['distances'][0]
            
            select_chunks = []
            for chunk, distance in zip(retrieved_text, distances):
                keyword_score = 0
                chunk_words = chunk.lower().split()

                for word in query_words:
                    if word in chunk_words:
                        keyword_score +=1

                final_score = keyword_score - distance
                select_chunks.append(
                    (final_score, chunk)
                )

            select_chunks = sorted(select_chunks, reverse=True)
            select_chunks = select_chunks[:2]
            top_chunks = []
            for final_score, chunk in select_chunks:
                print(f"Final Score: {final_score} | ")
                print(f"Chunk: {chunk}")
                top_chunks.append(chunk)

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
        else:
            reject()

    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail=  str(e)
        )