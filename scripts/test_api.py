import requests
import os

url = "http://localhost:8000/search/text"
headers = {"x-api-key": os.getenv("API_KEY", "change-me")}
payload = {"query": "리트리버", "topk": 20}

resp = requests.post(url, headers=headers, json=payload)
print(resp.json())
