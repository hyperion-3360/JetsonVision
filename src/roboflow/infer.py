import base64
from multiprocessing.pool import ThreadPool
import threading
import numpy as np
import requests
from io import BytesIO
from PIL import Image

class Roboflow2024:
    task = "object_detection"
    api_key = "IHApyUdPDP5QFjMBWv7k"

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    params = {'api_key': api_key}

    def __init__(self, server="http://localhost", port=9001, model_id="frc-2024-team-3360/3", pool=True):
        """
        Init client for roboflow inference server
        @param server:  ip address of inference server. Default is `http://localhost`
        @param port:    port of the inference server. Default is `9001`
        @param model_id: model_id of the roboflow model
        @param pool:    Should async inference request (#infer_async) use a thread pool of 8 threads or launch a new thread for every request ?
        """
        self.server = f'{server}:{port}'
        self.model_id = model_id

        # TODO: with multiple threads, do we risk having an inference complete before an earlier request ?
        self._req_pool = ThreadPool(processes=1) if pool else None

    def shutdown(self):
        print("Shutting down ai requests")
        if self._req_pool is not None:
            self._req_pool.terminate()

    @staticmethod
    def _encode(image: np.ndarray):
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


    def infer_async(self, image:np.ndarray, callback, model_class="note"):
        """
        Run inference on an image and pass the inference result to the callback method
        The passed data will be either None or the biggest bounding box center of the wanted class,
        with normalized coordinates relative to the top-left corner of the image / camera fov
        """
        url = f'{self.server}/{self.model_id}'

        if self._req_pool is not None:
            self._req_pool.apply_async(Roboflow2024._infer_async, args=(url, image, model_class, callback))
        else:
            threading.Thread(
                target=Roboflow2024._infer_async,
                args=(url, image, model_class, callback)
            ).start()


    def infer(self, image: np.ndarray, model_class="note"):
        """
        Run inference on an image and return the biggest bounding box center of the wanted class
        with normalized coordinates relative to the top-left corner of the image / camera fov
        @param image:       image on which to run the ai object detection model on
        @param model_class: which class to parse for
        @returns None if no bb of that class was detected
        """
        return Roboflow2024._infer(
                f'{self.server}/{self.model_id}',
                image,
                model_class
            )

    @staticmethod
    def _infer(url: str, image: np.ndarray, model_class: str):
        """
        Run inference on an image and return the biggest bounding box center of the wanted class
        with normalized coordinates relative to the top-left corner of the image / camera fov
        @param image:       image on which to run the ai object detection model on
        @param model_class: which class to parse for
        @returns None if no bb of that class was detected
        """
        data = Roboflow2024._encode(image)

        try:
            response = requests.post(
                            url,
                            params=Roboflow2024.params,
                            headers=Roboflow2024.headers,
                            data=data
                        ).json()

            note = Roboflow2024._parse(response, model_class)

            if note is None:
                return None

            return Roboflow2024._convert_bb_to_rel(note, (int(response['image']['width']), int(response['image']['height'])))

        except Exception as e: # Roboflow container is down ?
            print(str(e))
            return None

    @staticmethod
    def _infer_async(url: str, image: np.ndarray, model_class: str, callback):
        callback(
            Roboflow2024._infer(url, image, model_class)
        )

