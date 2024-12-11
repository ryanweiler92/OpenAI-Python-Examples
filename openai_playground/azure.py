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

    def azure_request(self):
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
