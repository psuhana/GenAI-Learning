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

REWRITE_QUERY_PROMPT = """ You are an AI Rewiter for AI/RAG Assistance.
                    Rewrite user queries into short retrieval friendly search queries.
                    Rules:
                    - Keep queries concise.
                    - Preserve conversational meaning.
                    - Preserve AI/RAG domain context.
                    - DO NOT answer the question
                    - DO NOT explain concepts
                    - Return only the rewritten query.
                    """

rewritten_response = client.chat.completions.create(
    model= 'gpt-4.1-mini',
    messages= [
        {
            'role':'system',
            'content': REWRITE_QUERY_PROMPT
        },
        {
            'role':'user',
            'content': query
        }
    ]
)

rewritten_query = rewritten_response.choices[0].message.content

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

retrieved_context = []
for i, text in enumerate(retrieved_text):
    chunk_id = result['metadatas'][0][i]['chunk_id']
    retrieved_context.append(
        f"[Chunk ID: {chunk_id}] | {text}"
    )

print(retrieved_context)

chat_response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages=[
        {
            "role":"system",
            "content": f"""
                            You are an AI Assistant.
                            Answer ONLY using the retrieved context.
                            For every factual statement,
                            include the supporting chunk id in this format:
                            [Chunk: <id>]

                            Example:
                            Embeddings convert text into vectors. [Chunk: 4]
                            If multiple chunks support a statement:
                            [Chunk: 4,7]
                            Retrieved Context:
                            {retrieved_context}
                            """
        },
        {
            "role":"user",
            "content":query
        }
    ]
)

chat_reply = chat_response.choices[0].message.content

print(chat_reply)