from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
chroma_client = chromadb.PersistentClient(
    path = r"C:\Users\suhan\Desktop\LLM\chroma_storage"
)

# INGESTION

child_collection = chroma_client.get_or_create_collection(
    name= 'child_db'
)

count_collection = child_collection.count()

if os.path.exists('notes_parent_child.txt'):
        with open('notes_parent_child.txt', 'r') as file:
            content = file.read()
            parents = content.split('\n\n') 
else:
     print('File not found!')

if count_collection > 0:
    print('Collection exists in DB')
else: 
    child_counter = 0

    for parent_id, parent in enumerate(parents):
        children = parent.split('. ')
        for chunk in children:
            chunk = chunk.strip()
            chunk_response = client.embeddings.create(
                model = 'text-embedding-3-small',
                input= chunk
            )
            chunk_embed = chunk_response.data[0].embedding

            child_collection.add(
                ids=[str(child_counter)],
                documents=[chunk],
                embeddings=[chunk_embed],
                metadatas=[
                    {
                        'parent_id': parent_id,
                        'chunk_id': child_counter
                    }
                ]
            )

            child_counter += 1
    print('Successfully added collection')
    print(f"Collection Count: {child_collection.count()}")

# RETRIEVAL

query = input("Ask a Question: ")
rewritten_response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages=[
        {
            'role':'system',
            'content': """ You are an AI Rewiter for AI/RAG Assistance.
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
            'role':'user',
            'content': query
        }
    ]
)
rewritten_query = rewritten_response.choices[0].message.content

query_response = client.embeddings.create(
    model='text-embedding-3-small',
    input= rewritten_query
)
query_embed = query_response.data[0].embedding

results = child_collection.query(
    query_embeddings=[query_embed],
    n_results= 4
)

retrieved_text = results['documents'][0]
child_ids = results['metadatas'][0]
print(f"Child Ids: {child_ids}")

retrieved_parent_ids = []
for i, chunk in enumerate(retrieved_text):
    retrieved_parent_id = results['metadatas'][0][i]['parent_id']
    retrieved_parent_ids.append(retrieved_parent_id)

final_retrieved_ids = list(set(retrieved_parent_ids))

texts = []
for p_id in final_retrieved_ids:
    paragraph = parents[p_id]
    texts.append(paragraph)

combined_context = '\n'.join(texts)

compressed_response = client.chat.completions.create(
     model = 'gpt-4.1-mini',
     messages= [
          {
               'role':'system',
               'content': f"""You are a context compressor.
                        
                        Given a user query and retrieved text,
                        extract only the information relevant
                        to answering the query.
                        This is the retrieved text: {combined_context}

                        Return only relevant information."""
          },
          {
               'role':'user',
               'content': query
          }
     ]
)
compressed_reply = compressed_response.choices[0].message.content
print(f"Compressed Reply: {compressed_reply}")

chat_response = client.chat.completions.create(
    model= 'gpt-4.1-mini',
    messages=[
        {
            'role':'system',
            'content': f"""You are AI Assistant.
                        Answer user's queries using the context of the retreived content below.
                        Retrieved Text: {compressed_reply}
                        Answer ONLY using retrieved context.
                        If answer cannot be found in retrieved context,
                        reply exactly:
                        "I don't know."""
        },
        {
            'role':'user',
            'content':query
        }
    ]
)
chat_reply = chat_response.choices[0].message.content

print(chat_reply)