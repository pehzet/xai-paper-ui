import base64
from openai import OpenAI
from config import OPENAI_API_KEY




client = OpenAI(
    api_key=OPENAI_API_KEY
)
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Generate a List of 3 Desserts with chocolate"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")