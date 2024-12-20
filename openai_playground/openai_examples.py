import os
from openai import OpenAI
from dotenv import load_dotenv
from openai_playground.image import url_to_base64
import pprint

load_dotenv()


class OpenAIExamples:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_KEY")

    def vision(self, prompt: str, url: str):
        client = OpenAI(
            api_key=self.api_key,
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
                model="gpt-4o",
            )
            return response

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    openai = OpenAIExamples()
    response = openai.vision(
        "<CAPTION>",
        "https://d2x51gyc4ptf2q.cloudfront.net/content/uploads/2023/04/10140915/Harry-Maguire-Victor-Lindelof-Man-Utd-F365.jpg",
    )
    pprint.pprint(response.choices[0].message.content)
