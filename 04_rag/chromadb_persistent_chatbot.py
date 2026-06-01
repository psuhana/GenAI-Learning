from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
import json

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
messages = []

client = OpenAI(api_key= api_key)
chroma_client = chromadb.PersistentClient(
    path= 'chroma_storage'
)
     
# INGESTION PHASE

if os.path.exists('memory_rag.json'):
     with open('memory_rag.json', 'r') as file:
          messages = json.load(file)

if os.path.exists('notes2.txt'):
    with open('notes2.txt', 'r') as file:
        content = file.read()
        pieces = content.split('.')

        collection = chroma_client.get_or_create_collection(
            name= "notes2_db"
        )

        count = collection.count()

        if count == 0:
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
                    ids = [str(i)],
                    documents= [piece],
                    embeddings= [embed],
                    metadatas= [
                        {
                            "source": "notes2.txt",
                            "chunk_id": i
                        }
                    ]
                )
        else:
            print('Collection exists.\n')

# RETRIEVAL PHASE
is_true = True

while is_true:
        query = input("STOP to STOP\nAsk Question: ")
        if query.lower() == "stop":
             break

        query_response = client.embeddings.create(
            model= 'text-embedding-3-small',
            input= query
        )

        query_embed = query_response.data[0].embedding

        result = collection.query(
            query_embeddings= [query_embed],
            n_results= 3
        )
        retrieved_text = result['documents'][0]
        metadata_text = result['metadatas'][0]

        combined_text = "\n".join(retrieved_text)

        print(metadata_text)
        print(f"Retrieved Text: \n{combined_text}\n")

        messages.append(
                {
                    "role":"system", "content": f"""You are AI assistant. Answer the user's questions using
                                                    the retrieved context given below.\n
                                                    Retrieved context: {combined_text}
                                                If answer not present in retrieved context
                                                then say you dont know."""
                }
        )
        messages.append(
              {
                    "role":"user", "content": query
                }
        )

        chat_response = client.chat.completions.create(
            model= 'gpt-4.1-mini',
            messages= messages
        )
        reply = chat_response.choices[0].message.content
        messages.append(
             {
                  "role":"assistant",
                  "content": reply
             }
        )
        if len(messages) > 12:
             messages = [messages[0]]+messages[-11:]

        usage = chat_response.usage

        print(f"Answer: {reply}\n\n")
        print(
            f"""-------------------------------
            Prompt Token: {usage.prompt_tokens}\n
            Completion Token: {usage.completion_tokens}\n
            Total Token: {usage.total_tokens}"""
        )

        with open('memory_rag.json', 'w') as file:
             json.dump(messages, file, indent = 4)