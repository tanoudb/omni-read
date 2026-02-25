from google.genai import Client

client = Client(api_key="AIzaSyCKNCVWaq5skUtF8Jw_izvScNwwzuJgPH8")  # Mets ta clé ici

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