import base64
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image


task = "object_detection"
model_id = "frc-2024-team-3360/3"
api_key = "IHApyUdPDP5QFjMBWv7k"


import requests

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}

params = {
    'api_key': 'IHApyUdPDP5QFjMBWv7k',
}


image = cv2.imread("./image.png")
image = Image.fromarray(np.uint8(image)).convert('RGB')

buffered = BytesIO()
image.save(buffered, quality=100, format="PNG")
img_str = base64.b64encode(buffered.getvalue())
data = img_str.decode("ascii")

response = requests.post('http://localhost:9001/frc-2024-team-3360/3', params=params, headers=headers, data=data)

print(response)
print(response.json())
