import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def phi3_local():
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
        base_url="http://localhost:8888/api",
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
