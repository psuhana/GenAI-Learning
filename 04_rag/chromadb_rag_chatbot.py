from openai import OpenAI
import os
import chromadb

api_key = os.getenv('OPENAI_API_KEY')

if api_key == None:
    print('API Key not found!')
else:
    client = OpenAI(api_key= api_key)

chroma_client = chromadb.Client()

if os.path.exists(r'C:\Users\suhan\Desktop\LLM\notes2.txt'):
    with open(r'C:\Users\suhan\Desktop\LLM\notes2.txt', 'r') as file:
        content = file.read()
        pieces = content.split('.')

    collection = chroma_client.get_or_create_collection(
        name= "notes2"
    )

    for i in range(len(pieces)):
        piece = pieces[i].strip()

        if piece == "":
            continue

        response = client.embeddings.create(
            model= 'text-embedding-3-small',
            input= piece
        )

        embed = response.data[0].embedding

        collection.add(
            ids=[str(i)],
            documents= [piece],
            embeddings= [embed]
        )

query = input('Ask Question: ')
query_response = client.embeddings.create(
    model= 'text-embedding-3-small',
    input= query
)

query_embed = query_response.data[0].embedding

results = collection.query(
    query_embeddings= [query_embed],
    n_results= 1
)

retrieved_result = results['documents'][0][0]

chat_response = client.chat.completions.create(
    model= 'gpt-4.1-mini',
    messages= [
        {
            "role":"system", "content": f"""You are an AI assistant.
                Answer the user's question using the retrieved context below.
                Retrieved Context:
                {retrieved_result}

            If answer is not present in retrieved context,
            say you don't know."""
        },
        {
            "role": "user", "content": query
        }
    ]
)

output = chat_response.choices[0].message.content
print(output)

chroma_client.delete_collection("notes2")