import os
import google.generativeai as genai

# Set your key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAs7V95vHZW0gINr9tnJOs9yBXR00M9DjA"
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# List all models that support generating content
print("--- Available Models for Your API Key ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")