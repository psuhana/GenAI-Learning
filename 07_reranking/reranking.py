from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import chromadb

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key= api_key)
chroma_client = chromadb.PersistentClient(
    path='chroma_storage'
)

messages = []
is_true = True

# INGESTION

collection = chroma_client.get_or_create_collection(
    name='notes2_db'
)

count = collection.count()

if count > 0:
    print("Collection Exists in DB.")
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
                documents=[chunk],
                embeddings=[chunk_embed],
                metadatas=[
                    {
                        "source": "notes2.txt",
                        "chunk_id": i
                    }
                ]
            )

# RETRIEVAL

if os.path.exists('memory_chunking.json'):
    with open('memory_chunking.json', 'r') as file:
        messages = json.load(file)

while is_true:
    query = input('Stop to Stop\nAsk a Question: ')
    if query.lower() == 'stop':
        break

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
            }
        ] + messages[-4:] + [
            {
                "role":"user",
                "content":query
            }
        ]
    )
    rewritten_text = rewritten_response.choices[0].message.content
    print("Rewritten: ", rewritten_text)

    query_response = client.embeddings.create(
        model='text-embedding-3-small',
        input= rewritten_text
    )
    query_embed = query_response.data[0].embedding

    # RERANKING

    query_words = rewritten_text.lower().split()
    keyword_score = 0

    results = collection.query(
        query_embeddings= [query_embed],
        n_results= 10
    )

    result_text = results['documents'][0]
    distances = results['distances'][0]

    scored_chunks = []

    for chunk, distance in zip(result_text, distances):
        keyword_score = 0
        chunk_words = chunk.lower().split()
        for word in query_words:
            if word in chunk_words:
                keyword_score +=1

        final_score = keyword_score - distance
        scored_chunks.append(
            (final_score, chunk, distance, keyword_score)
        )
        
    scored_chunks = sorted(scored_chunks, reverse=True)
    scored_chunks = scored_chunks[:2]

    top_chunks = []
    
    for s_finalscore, s_chunk, s_distance, s_keyword_score in scored_chunks:
        print(
                f"Chunk: {s_chunk} | "
                f"Distance: {s_distance} "
                f"Keyword Score: {s_keyword_score} "
                f"Final Score: {s_finalscore} "
            )
        top_chunks.append(s_chunk)

    count_chunks = len(scored_chunks)

    if count_chunks > 0:
        combined_text = '\n'.join(top_chunks)

        messages.append(
            {
                "role":"user",
                "content": query
            }
        )
        temp_message = [
            {
            "role":"system",
            "content": f"""You are AI Assistant.
                            Answer user's queries using the context of the retreived content below.
                            Retrieved Text: {combined_text}
                            Answer ONLY using retrieved context.
                            If answer cannot be found in retrieved context,
                            reply exactly:
                            "I don't know."""
            }
        ] + messages
        
        chat_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages= temp_message
        )
        chat_reply = chat_response.choices[0].message.content

        print("Answer: ",chat_reply)

        messages.append(
            {
                "role":"assistant",
                "content": chat_reply
            }
        )

        if len(messages)>12:
            messages = messages[-12:]

        with open('memory_chunking.json','w') as file:
            json.dump(messages, file, indent = 4)

        usage = chat_response.usage

        print('Total tokens used: ', usage.total_tokens)
    else:
        print("Query out of context!")