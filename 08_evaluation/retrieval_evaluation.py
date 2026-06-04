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

evaluation_questions = [
    {
        "question": "What is ChromaDB?",
        "expected_chunk": "ChromaDB"
    },

    {
        "question": "What is SQL used for?",
        "expected_chunk": "SQL"
    },

    {
        "question": "What are embeddings?",
        "expected_chunk": "Embeddings"
    },

    {
        "question": "What is prompt engineering?",
        "expected_chunk": "Prompt engineering"
    }
]

if os.path.exists('memory_chunking.json'):
    with open('memory_chunking.json', 'r') as file:
        messages = json.load(file)

hits = 0
count_ques = 0

for item in evaluation_questions:
    query = item['question']
    print("Query: ", query)

    expected_chunk = item['expected_chunk']
    print('Expected Chunk: ',expected_chunk)

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
                f"Chunk: {s_chunk} \n"
                f"Distance: {s_distance} \n"
                f"Keyword Score: {s_keyword_score} \n"
                f"Final Score: {s_finalscore} \n"
            )
        top_chunks.append(s_chunk)

    found = False

    for check_chunk in top_chunks:
        if expected_chunk.lower() in check_chunk.lower():
            found = True
            break

    if found:
        hits+=1
        print('PASS')
    else:
        print('FAIL')

    count_ques +=1

accuracy = hits/count_ques * 100

print('Total Questions: ', count_ques)
print('Total Hits: ', hits)
print('Accuracy: ', accuracy)