from huggingface_hub import HfApi
import re

# Initialize the API
api = HfApi()

# Define the maximum number of parameters (15M in this case)
MAX_PARAMS = 15 * 10**6


# Function to extract the number of parameters from the metadata
def extract_params(model_info):
    metadata = {}

    if model_info.config:
        metadata = model_info.config.get("metadata", {})

    if not metadata and model_info.card_data:
        metadata = model_info.card_data.get("metadata", {})

    # Handle variations in how parameters might be reported
    num_params_str = (
        metadata.get("num_params")
        or metadata.get("model_size_params")
        or metadata.get("parameters")
        or "0"
    )

    # Clean and convert the parameter string to an integer
    num_params_str = re.sub(r"[^0-9.]", "", num_params_str)
    try:
        num_params = float(num_params_str)
        if "M" in num_params_str.upper() or "m" in num_params_str.lower():
            num_params *= 10**6
        elif "B" in num_params_str.upper() or "b" in num_params_str.lower():
            num_params *= 10**9
    except ValueError:
        num_params = 0

    return num_params


# List all models (this may take a while and return a large number of models)
all_models = api.list_models(limit=30)

filtered_models = []
for model in all_models:
    repo_id = model.id  # Extract the repo_id from the ModelInfo object

    # Fetch model info with error handling
    try:
        model_info = api.model_info(repo_id=repo_id)
    except Exception as e:
        print(f"Failed to retrieve info for {repo_id}: {e}")
        continue  # Skip this model if there's an error

    num_params = extract_params(model_info)
    if num_params < MAX_PARAMS:
        filtered_models.append(repo_id)
    else:
        print(f"Skipping {repo_id} with {num_params} parameters")  # Debug info

print("Models with less than 15 million parameters:")
for model in filtered_models:
    print(model)
