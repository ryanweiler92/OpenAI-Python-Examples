import requests
from pprint import pprint


def get_model_info(model_id):
    # Main model card info endpoint
    base_url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(base_url)
    model_info = response.json()

    # Key fields to check:
    # 1. model_info['config'] - Contains architecture details
    #    - hidden_size
    #    - num_attention_heads
    #    - num_hidden_layers
    #    These help calculate total parameters

    # 2. model_info['cardData'] - Contains model card parsed data
    #    - Model size is often listed here
    #    - GPU requirements if specified

    # 3. File list to check weights size
    files_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
    files = requests.get(files_url).json()
    # Look for .bin files and their sizes

    return {
        "config": model_info.get("config"),
        "card_data": model_info.get("cardData"),
        "files": files,
    }


# Usage
model_id = "nvidia/MambaVision-S-1K"
info = get_model_info(model_id)
pprint(info)
