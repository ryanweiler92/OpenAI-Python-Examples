import os
from openai import OpenAI
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


class StructuredResponses:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_KEY")
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

    def get_player_info_vllm_json(self):
        """Get structured response from vLLM model with JSON schema"""
        client = OpenAI(base_url=self.llama3_endpoint, api_key="EMPTY")

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": "Name a few Manchester United players you know with their positions, countries, and skill ratings.",
                },
            ],
            extra_body={"guided_json": self.json_schema},
        )

        return response.choices[0].message.content

    def get_player_info_vllm_pydantic(self):
        """Get structured response from vLLM model with Pydantic schema"""
        json_schema = PlayerInfo.model_json_schema()

        client = OpenAI(base_url=self.llama3_endpoint, api_key="EMPTY")

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": "Name a few Manchester United players you know with their positions, countries, and skill ratings.",
                },
            ],
            extra_body={"guided_json": json_schema},
        )

        return response.choices[0].message.content

    def get_player_info_openai_json(self):
        """Get structured response from OpenAI model with JSON schema"""

        client = OpenAI(api_key=self.api_key)

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

    def get_player_info_openai_pydantic(self):
        """Get structured response from OpenAI model with Pydantic schema"""

        client = OpenAI(api_key=self.api_key)

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Name a few Manchester United players you know with their positions, countries, and skill ratings.",
                },
            ],
            response_format=PlayerInfo,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    StructureClass = StructuredResponses()
    player_info = StructureClass.get_player_info_openai_pydantic()
    print(player_info)
