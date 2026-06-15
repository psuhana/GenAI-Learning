import os

def ingestion(client, knowledge_collection):
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
                input=chunk
            )
            chunk_embed = chunk_response.data[0].embedding
            knowledge_collection.add(
                ids=[str(i)],
                documents=[chunk],
                embeddings=[chunk_embed],
                metadatas=[
                    {
                        'source':'notes2.txt',
                        'chunk_id':i
                    }
                ]
            )
        
        print('Successfully added to Collection!')