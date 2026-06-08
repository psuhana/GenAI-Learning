from openai import OpenAI
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import os
import chromadb

def generate(response):
        for chunk in response:
            token = chunk.choices[0].delta.content

            if token:
                print(token, end="", flush= True)
                yield token

def rag_pipeline(question):

    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(
        path = r"C:\Users\suhan\Desktop\LLM\chroma_storage"
    )

    # INGESTION

    collection = chroma_client.get_or_create_collection(
        name = 'notes2_db'
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
                try:
                    chunk_response = client.embeddings.create(
                        model='text-embedding-3-small',
                        input= chunk
                    )
                    chunk_embed = chunk_response.data[0].embedding

                    collection.add(
                        ids= [str(i)],
                        documents=[chunk],
                        embeddings=[chunk_embed],
                        metadatas=[
                            {
                                "source":"notes2.txt",
                                "chunk_id": i
                            }
                        ]
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code= 500,
                        detail=str(e)
                    )
                
        print('Collection added to DB.')

    # RETRIEVAL
    
    try:
        query = question

        rewritten_response = client.chat.completions.create(
            model= 'gpt-4.1-mini',
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
            ]
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

        retrieved_text = results["documents"][0]
        distances = results["distances"][0]

        query_words = rewritten_reply.lower().split()

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

        select_chunks = sorted(select_chunks, reverse= True)
        select_chunks = select_chunks[:2]

        top_chunks =[]

        for final_score, chunk in select_chunks:
            print (f"Final Score: {final_score} | "
                f"Chunk: {chunk}")

            top_chunks.append(chunk)

        combined_text = "\n".join(top_chunks)

        chat_response = client.chat.completions.create(
            model = 'gpt-4.1-mini',
            messages= [
                {
                    "role":"system",
                    "content":f"""You are AI Assistant.
                                Answer user's queries in detailed using the context of the retreived content below.
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
            ],
            stream= True
        )

        return StreamingResponse(
            generate(chat_response),
            media_type= 'text/plain'
        )
    
    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail= str(e)
        )