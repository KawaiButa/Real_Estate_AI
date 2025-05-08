import os
import json
import pinecone

pc = pinecone.Pinecone(os.getenv("PINECONE_API_KEY"))

with open("property_embeddings.json", "r") as f:
    records = json.load(f)

index_name = "properties"
embed_dim = len(records[0]["embedding"])
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=embed_dim,
        metric="euclidean",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)
batch_size = 1000
for i in range(0, len(records), batch_size):
    batch = records[i : i + batch_size]
    batch = [{"id": item["id"], "values": item["embedding"]} for item in batch]
    index.upsert(vectors=batch, show_progress=True, batch_size=batch_size)
    print(f"Upserted vectors {i} {i+len(batch)-1}")

print("All embeddings loaded into Pinecone!")
