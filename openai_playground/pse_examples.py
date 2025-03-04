import os
import json
from openai import OpenAI
from transformers import AutoTokenizer
from dotenv import load_dotenv
from pse.structuring_engine import StructuringEngine


class PSEExamples:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_key")
        self.schema = {
            "type": "object",
            "properties": {
                "players": {
                    "type": "array",
                    "player": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "reason": {"type": "string"},
                            "ranking": {"type": "integer"},
                        },
                        "required": ["name", "reason", "ranking"],
                    },
                    "minItems": 5,
                    "maxItems": 5,
                },
            },
            "required": ["players"],
        }

        system_message = f"""
        Please only return pure JSON. You must follow this schema when generating your response:
        {json.dumps(self.schema, indent=2)}
        """
        prompt = "Who are the top 5 basketball players of all time"

        self.messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    def text(self, messages: list[dict]):
        client = OpenAI(api_key=self.api_key)

        try:
            respone = client.chat.completions.create(
                messages=messages,
                model="gpt-4o",
            )
            return respone
        except Exception as e:
            print(f"An error occurred: {e}")

    def validate_openai_pse(self, tokenizer_name: str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        engine = StructuringEngine(tokenizer=tokenizer)
        engine.configure_json(self.schema)

        response = self.text(messages=self.messages)

        raw_output = response.choices[0].message.content
        print("Raw_output:", raw_output)

        try:
            engine.consume_text(raw_output)
            parsed_output = engine.parse_structured_output()
            print("Parsed_output:", parsed_output)

            print("Accept_state:", engine.has_reached_accept_state)

            if engine.steppers:
                for i, stepper in enumerate(engine.steppers):
                    print(f"Stepper {i} state:", stepper.get_current_value())

            if engine.has_reached_accept_state:
                print("Valid JSON output:", parsed_output["players"][0])
                print("The GOAT of basketball is:", parsed_output["players"][0]["name"])
            else:
                print("JSON VALIDATION FAILED")
        except Exception as e:
            print(f"Error parsing output: {e}")


if __name__ == "__main__":
    pse = PSEExamples()
    pse.validate_openai_pse(
        "microsoft/phi-4"
    )  # The model can be any text-generation model
