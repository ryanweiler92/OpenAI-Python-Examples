import os
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv


class Player(BaseModel):
    name: str
    position: str
    country: str
    skill: int = Field(ge=1, le=10)


class PlayerInfo(BaseModel):
    players: List[Player]


class AzureResponses:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_KEY")
        self.azure_key = os.getenv("AZURE_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.llama3_endpoint = os.getenv("LLAMA3_ENDPOINT")
        self.json_schema = {
            "type": "object",
            "properties": {
                "players": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "position": {"type": "string"},
                            "country": {"type": "string"},
                            "skill": {"type": "integer"},
                        },
                        "required": ["name", "position", "country", "skill"],
                    },
                }
            },
            "required": ["players"],
        }

    def standard_request(self):
        client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_key,
            api_version="2024-10-21",
        )

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "When was Microsoft founded?"},
            ],
        )

        return completion.choices[0].message.content

    def get_structured_response_json(self):
        """Get structured response from OpenAI model with JSON schema"""
        client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_key,
            api_version="2024-10-21",
        )

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Name a few Manchester United players you know with their positions, countries, and skill ratings.",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "player_schema", "schema": self.json_schema},
            },
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    azure = AzureResponses()
    print(azure.get_structured_response_json())
