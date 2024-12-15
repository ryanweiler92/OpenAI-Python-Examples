import os
from openai import OpenAI
from dotenv import load_dotenv


class Embeddings:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_KEY")

    def get_local_embeddings(self):
        """Get local embeddings from the embeddings model"""
        try:
            client = OpenAI(api_key="NONE", base_url="http://localhost:8888/api")

            response = client.embeddings.create(
                model="text-embedding-v1",
                input=[
                    "Once upon a time",
                    "There was a frog",
                    "Who lived in the forest",
                ],
            )

            return response.data
        except Exception as e:
            return f"An error occurred: {e}"

    def get_openai_embeddings(self):
        """Get OpenAI embeddings from the embeddings model"""
        try:
            client = OpenAI(api_key=self.api_key)

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[
                    "Once upon a time",
                    "There was a frog",
                    "Who lived in the forest",
                ],
            )

            return response.data
        except Exception as e:
            return f"An error occurred: {e}"


if __name__ == "__main__":
    embeddings = Embeddings()
    print(embeddings.get_openai_embeddings())
