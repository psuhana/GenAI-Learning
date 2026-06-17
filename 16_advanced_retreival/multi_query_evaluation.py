from openai import OpenAI
from dotenv import load_dotenv
from ingestion import ingestion
import os
import chromadb

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path = r"C:\Users\suhan\Desktop\LLM\chroma_storage"
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

MULTI_QUERY_PROMPT = """
                    Generate 3 different retrieval-focused
                    search queries for the user's question.

                    Return one query per line.

                    Do not answer.
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

PASS = 0
question_num = 0

for item in eval_queries:
    question_num += 1

    query = item['query']
    expected = item['expected'].lower().strip()

    rewritten_response = client.chat.completions.create(
        model = 'gpt-4.1-mini',
        messages=[
            {
                'role':'system',
                'content': MULTI_QUERY_PROMPT
            },
            {
                'role':'user',
                'content': query
            }
        ]
    )

    rewritten_queries = rewritten_response.choices[0].message.content.split('\n')

    all_results = []
    for query in rewritten_queries:
        query_response = client.embeddings.create(
            model='text-embedding-3-small',
            input= query
        )
        query_embed = query_response.data[0].embedding

        result = knowledge_collection.query(
            query_embeddings= [query_embed],
            n_results= 2
        )
        retrieved_text = result['documents'][0]

        all_results.extend(retrieved_text)
        
    unique_chunks = list(set(all_results))
    print(f"Unique Chunks: {unique_chunks}\n")
    print(f"Expected: {expected}\n")
    hit = False
    for chunk in unique_chunks:
        if expected in chunk.lower().strip():
            hit = True
            PASS += 1
            break

    if hit:
        print(f'{query}: PASS')
    else:
        print(f'{query}: FAIL')

accuracy = (PASS/question_num)*100
print(f"PASSES: {PASS}")
print(f"Questions: {question_num}")
print(f"Accuracy: {accuracy}")