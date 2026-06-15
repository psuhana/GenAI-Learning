from openai import OpenAI
from dotenv import load_dotenv
from ingestion import ingestion
import os
import chromadb
import redis

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

def run_evals(prompt, version):
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
    total_words = 0

    for item in eval_queries:
        question_num += 1
        query = item['query']
        expected = item['expected'].lower().strip()

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
                        'content': prompt
                    },
                    {
                        'role':'user',
                        'content': query
                    }
                ]
            )
            rewritten_reply = rewritten_response.choices[0].message.content
            total_words = len(rewritten_reply.split())
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

    accuracy = (PASS/question_num)*100
    avg_words = total_words/question_num

    return {
        "version": version,
        "accuracy": accuracy,
        "avg words": avg_words
    }

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

PROMPTS = {
    'v1':REWRITE_QUERY_V1,
    'v2':REWRITE_QUERY_V2
}

results = []
for version, prompt in PROMPTS.items():
    result = run_evals(prompt, version)
    results.append(result)

winner = max(
    results,
    key = lambda x:(
        x['accuracy'],
        -x['avg words']
    )
)

print("======== A/B TEST RESULTS ========")
for result in results:
    print(f"Version: {result['version']}")
    print(f"Accuracy: {result['accuracy']}")
    print(f"Average Rewrite Length: {result['avg words']}")
    print()

print(f"Winner: {winner['version']}")
print('Reason: Equal accuracy, shorter average rewrite length')
