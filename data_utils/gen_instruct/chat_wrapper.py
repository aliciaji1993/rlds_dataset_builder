import base64
import cv2
import numpy as np
import io
import torch
from typing import Union, List

from abc import ABC, abstractmethod
from PIL import Image

from openai import OpenAI
from prismatic import load


class ChatWrapper(ABC):

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_message(self, **args):
        raise NotImplementedError


class ChatGPT(ChatWrapper):

    def __init__(self, model_name: str = "gpt-4o", system_prompt: str = "") -> None:
        self.chat = OpenAI()
        self.model_name = model_name
        self.system_prompt = system_prompt

    def _generate_image_content(self, image: Union[Image.Image, np.ndarray]):
        # convert image to BytesIO and encode in openai image format
        # image_bytes = io.BytesIO()
        # image.save(image_bytes, format="jpg", dpi=300)
        # image_bytes.seek(0)
        # base64_image = base64.b64encode(image_bytes.read()).decode("utf-8")
        image_arr = np.array(image) if type(image) is not np.ndarray else image
        encoded_image = cv2.imencode(".jpg", image_arr)[1]
        base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

        return dict(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        )

    def send_message(
        self,
        images: Union[Image.Image, np.ndarray, List[np.ndarray]],
        user_prompt: str,
        verbose=False,
    ):
        image_content = (
            [self._generate_image_content(m) for m in images]
            if isinstance(images, list)
            else [self._generate_image_content(images)]
        )

        completion = self.chat.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": image_content
                    + [
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            # response_format={"type": "json_object"},
        )
        response = completion.choices[0].message
        if verbose:
            print("Response: ", response.content)
        return response.content


class ChatVLM(ChatWrapper):

    def __init__(
        self,
        model_name: str,
        hf_token: str,
        system_prompt: str = "",
        device: torch.DeviceObjType = torch.device("cuda"),
    ) -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.system_prompt = system_prompt

        # init vlm
        self.vlm = load(model_name, hf_token=hf_token).to(
            self.device, dtype=torch.bfloat16
        )

    def send_message(
        self, image: Union[Image.Image, np.ndarray], user_prompt: str, verbose=True
    ):
        # Build prompt
        prompt_builder = self.vlm.get_prompt_builder(system_prompt=self.system_prompt)
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        # Generate!
        generated_text = self.vlm.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=512,
            min_length=1,
        )

        if verbose:
            print("Response: ", generated_text)

        return generated_text
