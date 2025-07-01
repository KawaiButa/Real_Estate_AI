import os
import pinecone
pc = pinecone.Pinecone(os.getenv("PINECONE_API_KEY"))
property_index = pc.Index("properties")
article_index = pc.Index("articles")