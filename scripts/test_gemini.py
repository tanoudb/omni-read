import os

from google.genai import Client

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY non défini (voir .env)")

client = Client(api_key=api_key)

success = 0

for i in range(30):
    try:
        response = client.models.generate_content(
            model="gemini-2.5.flash",
            contents="test"
        )
        print(f"Requête {i+1}: OK -> {response.text}")
        success += 1
    except Exception as e:
        print(f"Requête {i+1}: ERREUR -> {e}")
        break

print(f"\nTotal de requêtes réussies: {success}")
