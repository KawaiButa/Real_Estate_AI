import os
from dotenv import load_dotenv
from supabase import create_client, Client
load_dotenv()

def provide_supabase_client() -> Client:
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    return create_client(url, key)