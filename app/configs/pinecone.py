import os
from chromadb.config import Settings
import pinecone
pc = pinecone.Pinecone(os.getenv("PINECONE_API_KEY"))
property_index = pc.Index("properties")