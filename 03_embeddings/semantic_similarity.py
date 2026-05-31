from openai import OpenAI
import os
import numpy as np

api_key = os.getenv('OPENAI_API_KEY')
if api_key == None:
    print('No Key Found')
else:
    client = OpenAI(api_key= api_key)

sentences = [
    "Machine learning is powerful",
    "Aritificial Intelligence is amazing",
    "I love pizza"
]

embeddings =[]

for sentence in sentences:
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input= sentence
    )
    embed = response.data[0].embedding
    embeddings.append(embed)

def cosine_similarity(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    num = np.dot(vec1,vec2)
    den1 = np.linalg.norm(vec1)
    den2 = np.linalg.norm(vec2)
    return num/ (den1 * den2)


similarity1 = cosine_similarity(
    embeddings[0],
    embeddings[1]
)
similarity2 = cosine_similarity(
    embeddings[0],
    embeddings[2]
)

print(f"Similarity 1: {similarity1}\nSimilarity 2: {similarity2}")