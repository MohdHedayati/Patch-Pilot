import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
try:
    print("Sending ping...")
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Respond with exactly one word: "Alive".'
    )
    print("Response:", response.text)
except Exception as e:
    print("Crash:", e)