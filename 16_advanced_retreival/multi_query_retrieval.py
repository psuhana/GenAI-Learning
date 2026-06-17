from openai import OpenAI
from dotenv import load_dotenv
from ingestion import ingestion
import chromadb
import os

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key= api_key)

chroma_client = chromadb.PersistentClient(
        path = r"C:\Users\suhan\Desktop\LLM\chroma_storage"
    )

collection = chroma_client.get_or_create_collection(
    name= 'notes2_db'
)
count_collection = collection.count()

if count_collection == 0:
    ingestion(collection)
else:
    print('Collection exists in DB')

query = input('Ask a question: ')

MULTI_QUERY_PROMPT = """
                    Generate 3 different retrieval-focused
                    search queries for the user's question.

                    Return one query per line.

                    Do not answer.
                    """

rewritten_response = client.chat.completions.create(
    model= 'gpt-4.1-mini',
    messages= [
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
print(f"Rewritten queries: {rewritten_queries}")

all_results = []
for query in rewritten_queries:
    query_response = client.embeddings.create(
        model='text-embedding-3-small',
        input= query
    )
    query_embed = query_response.data[0].embedding

    result = collection.query(
        query_embeddings= [query_embed],
        n_results= 2
    )
    retrieved_text = result['documents'][0]
    all_results.extend(retrieved_text)

print(f"All Results: {all_results}")

unique_chunks = list(set(all_results))
print(f"Unique Chunks: {unique_chunks}")

combined_text = '\n'.join(unique_chunks)

chat_response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages=[
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
    ]
)

chat_reply = chat_response.choices[0].message.content

print(chat_reply)