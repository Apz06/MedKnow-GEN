import requests

# Check ChEMBL response format
print("=== ChEMBL Response ===")
url = "https://www.ebi.ac.uk/chembl/api/data/mechanism?molecule_pref_name=Gefitinib&format=json&limit=3"
r = requests.get(url, timeout=20)
print(r.json())

import requests

print("=== MyDisease.io Response ===")
url = "https://mydisease.info/v1/query?q=breast+cancer&fields=all&size=1"
r = requests.get(url, timeout=20)
import json
print(json.dumps(r.json()["hits"][0], indent=2))