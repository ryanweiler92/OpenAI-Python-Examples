import requests
import base64


def url_to_base64(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        base64_string = base64.b64encode(response.content).decode("utf-8")

        return f"data:image/{response.headers['Content-Type'].split('/')[-1]};base64,{base64_string}"

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None
