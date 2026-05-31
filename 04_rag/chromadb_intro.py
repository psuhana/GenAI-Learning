from openai import OpenAI
import chromadb
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name = "my_notes"
)
text = "RAG combines retrieval with language models"

response = client.embeddings.create(
    model= "text-embedding-3-small",
    input= text
)

embed = response.data[0].embedding

collection.add(
    ids=["1"],
    documents= [text],
    embeddings= [embed]
)

query = input('Ask Question: ')
query_response = client.embeddings.create(
    model= "text-embedding-3-small",
    input= query
)

query_embedding = query_response.data[0].embedding

results = collection.query(
    query_embeddings= [query_embedding],
    n_results= 1
)

print(results["documents"])