import base64
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

class Roboflow2024:
    task = "object_detection"
    api_key = "IHApyUdPDP5QFjMBWv7k"

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    params = {'api_key': api_key}

    def __init__(self, server="http://localhost", port=9001, model_id="frc-2024-team-3360/3"):
        self.server = f'{server}:{port}'
        self.model_id = model_id

    @staticmethod
    def _encode(image: np.ndarray[float]):
        image = Image.fromarray(np.uint8(image)).convert('RGB')

        buffered = BytesIO()
        image.save(buffered, quality=100, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str.decode("ascii")

    @staticmethod
    def _parse(inference_result, model_class):
        """
        Select the biggest bounding box for a note
        """
        bbsize = lambda pred: int(pred['width'])*int(pred['height'])
        selected_pred = None

        for prediction in inference_result['predictions']:
            if prediction['class'] == model_class:
                if selected_pred is None or bbsize(prediction) > bbsize(selected_pred):
                    selected_pred = prediction

        return selected_pred


    @staticmethod
    def _convert_bb_to_rel(note, frame_size):
        """
        Convert a note's bounding box center to a relative value with the frame's top-left corner as the origin (open-cv coordinates)
        """
        w, h = frame_size
        return (note['x'] / w, note['y'] / h)


    def infer(self, image: np.ndarray[float], model_class="note"):
        """
        Run inference on an image and return the biggest bounding box center of the wanted class
        with normalized coordinates relative to the top-left corner of the image
        Returns None if no bb of that class was detected
        """
        data = Roboflow2024._encode(image)

        try:
            response = requests.post(
                            f'{self.server}/{self.model_id}',
                            params=Roboflow2024.params,
                            headers=Roboflow2024.headers,
                            data=data
                        )

            note = Roboflow2024._parse(response.json(), model_class)

            if note is None:
                return None

            return Roboflow2024._convert_bb_to_rel(note, (int(response['image']['width']), int(response['image']['height'])))

        except Exception as e: # Roboflow container is down ?
            print(str(e))
            return None

