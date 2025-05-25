from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
