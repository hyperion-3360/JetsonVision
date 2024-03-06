import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

model_id = "frc-2024-team-3360/2"
image_url = "https://media.roboflow.com/inference/soccer.jpg"

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)


results = client.infer(image_url, model_id=model_id)
print(results)
