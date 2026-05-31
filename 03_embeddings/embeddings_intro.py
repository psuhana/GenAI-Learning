from openai import OpenAI
import os

api_key = os.getenv('OPENAI_API_KEY')
if api_key == None:
    print('No Key Found')
else:
    client = OpenAI(api_key= api_key)

response = client.embeddings.create(
    model='text-embedding-3-small',
    input="I love Pizza"
)

embeddings = response.data[0].embedding
length = len(embeddings)
print(f'Number: {length}\n')
print(embeddings[:2])