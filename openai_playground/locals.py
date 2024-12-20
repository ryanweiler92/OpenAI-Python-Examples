import os
from openai import OpenAI
from dotenv import load_dotenv
from openai_playground.image import url_to_base64
import pprint

load_dotenv()


class LocalModels:
    def __init__(self):
        load_dotenv()
        self.base_url = "http://localhost:8888/api"

    def florence_vision_local(self, prompt: str, url: str):
        client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
        )

        try:
            base64_image = url_to_base64(url)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_image}},
                        ],
                    },
                ],
                model="florence",
            )
            return response

        except Exception as e:
            print(f"An error occurred: {e}")

    def phi3_local(self):
        client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
        )

        try:
            response = client.chat.completions.create(
                model="phi-3-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
            )
            print(response.choices[0].text)

        except Exception as e:
            print(f"An error occurred: {e}")

    def florence_vision_stream_local(self, prompt: str, url: str):
        client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
        )

        try:
            base64_image = url_to_base64(url)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_image}},
                        ],
                    },
                ],
                model="florence",
                stream=True,
            )

            full_response = ""

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content

            return full_response

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    image_response = LocalModels().florence_vision_stream_local(
        "<CAPTION>",
        "https://d2x51gyc4ptf2q.cloudfront.net/content/uploads/2023/04/10140915/Harry-Maguire-Victor-Lindelof-Man-Utd-F365.jpg",
    )
    pprint.pprint(image_response)
