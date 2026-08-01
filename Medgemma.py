from transformers import pipeline
from PIL import Image
import requests,io
import torch
import base64
import io

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    dtype=torch.bfloat16
    #device_map="auto",
)
def retrieve_information(image_url):
    # response = requests.get(
    #     image_url,
    #     headers={"User-Agent": "Mozilla/5.0"}
    # )
    # response.raise_for_status()

    image = Image.open(image_url).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this X-ray"}
            ]
        }
    ]

    output = pipe(
        text=messages,
        max_new_tokens=64
    )

    print(output[0]["generated_text"][-1]["content"])




if __name__ == "__main__":
    output = retrieve_information("x-ray_chest_for_test.jpg")#("https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png")
    print(output)