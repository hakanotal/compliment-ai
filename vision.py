from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import cv2

MODEL_ID = "unsloth/Llama-3.2-11B-Vision-Instruct"
DESCRIBE_PROMPT = "Describe the style and outfit of the person in the image."
COMPLIMENT_PROMPT = "Write a two-lines compliment about the person's outfit in this image. Make it fun and friendly."


class Vision2Text:

    def __init__(self):

        self.model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

        # Read a frame from the video capture
        ret, frame = cap.read()

        if ret:
            cv2.imwrite('image.jpg', frame)
            print("Image saved as 'image.jpg'")
        else:
            print("Failed to read a frame")

        cap.release()

        return Image.open("./image.jpg")   


    def generate_description(self, image):

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": DESCRIBE_PROMPT}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=128)
        image_description = self.processor.decode(output[0]).split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0]

        return image_description.strip()
    

    def generate_compliment(self, image, image_description):

        compliment_messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": DESCRIBE_PROMPT}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": image_description + '<|eot_id|>'}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": COMPLIMENT_PROMPT}
            ]},
        ]
        input_text = self.processor.apply_chat_template(compliment_messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=128)
        compliment = self.processor.decode(output[0]).split('<|start_header_id|>assistant<|end_header_id|>')[2].split('<|eot_id|>')[0]

        return compliment.strip()
