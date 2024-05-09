import json

def hf_token() -> str:
    with open('.tokens.json') as f:
        data = json.load(f)
    return data["huggingface"]
