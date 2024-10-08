import json
import numpy as np

from enum import IntEnum
from openai import OpenAI
from pathlib import Path
from PIL import Image
from typing import Union, List

import matplotlib.pyplot as plt

from .chat_wrapper import ChatWrapper


class InstructType(IntEnum):
    FREE_FORM = 0
    MAIN_DIRECT_4 = 1
    MAIN_DIRECT_8 = 2
    FORMAT_ACTION = 3


class ContextType(IntEnum):
    OBS_1_ACTIONS_MAP = 0
    OBS_1_ACTIONS_STRING = 1
    OBS_8_ACTIONS_STRING = 2


class ReasoningType(IntEnum):
    REASON_BY_ACTIONS = 0
    REASON_BY_SCENES = 1
    REASON_BY_STEPS = 2


DOWN_SAMPLE_KEYWORDS = [
    "move forward",
]

MAX_FILE_NAME_CHAR = 255

DEFAULT_INSTRUCTION = "continue the trajectory"


def plot_actions(ax, actions, color="b"):
    lim = np.max(np.abs(actions))
    # switch x and y axis, as x represents forward movement in real world
    ax.plot(-actions[:, 1], actions[:, 0], f"{color}o")  # 'o' means circles
    ax.plot(-actions[:, 1], actions[:, 0], f"{color}-")  # '-' means solid line
    # mark start spot as green and end spot as red
    ax.plot(-actions[0, 1], actions[0, 0], f"go")  # 'go' means green circles
    ax.plot(-actions[-1, 1], actions[-1, 0], f"ro")  # 'go' means red circles
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(aspect=1)


def visualize_step(image, actions, instruction, reasoning, save_path):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(
        instruction,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=12,
        wrap=True,
    )
    # draw fig (obs + action plot) on canvas
    axs[0].imshow(image)
    axs[0].axis("off")
    plot_actions(axs[1], actions)
    plt.figtext(
        0.0,
        0.0,
        reasoning,
        wrap=True,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontsize=6,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()


def generate_instruction(
    chat: ChatWrapper,
    images: Union[List[np.ndarray], np.ndarray],
    actions: np.ndarray,
    generation_prompt: List[str],
    context_type: ContextType,
    save_path: Path = None,
    num_retries: int = 5,
):
    # generate instruction from VLMs
    if context_type == ContextType.OBS_1_ACTIONS_MAP:
        # draw fig (obs + action plot) on canvas
        fig, axs = plt.subplots(1, 2, figsize=(28, 28))
        # show observation image
        axs[0].imshow(images)
        axs[0].axis("off")
        # plot actions
        plot_actions(axs[1], actions)
        fig.canvas.draw()
        obs_and_action_map = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        # use instruction list as prompt directly
        user_prompt = generation_prompt
        generated_text = chat.send_message(obs_and_action_map, user_prompt)
        plt.close()
    else:
        # Feed action list and instruction list as text prompt
        y_mirrored_actions = actions * np.array([0, -1] * actions.shape[0]).reshape(
            actions.shape
        )
        user_prompt = (
            f"Given list of actions: {y_mirrored_actions}\n" + generation_prompt
        )
        num_tries = 0
        stop = False
        while num_tries < num_retries and not stop:
            try:
                bgr_images = (
                    [image[:, :, ::-1] for image in images]
                    if isinstance(images, list)
                    else images[:, :, ::-1]
                )
                generated_text = chat.send_message(bgr_images, user_prompt)
                reasoning = json.loads(generated_text)["reasoning"]
                instruction = json.loads(generated_text)["instruction"]
                stop = True
            except Exception as e:
                print("Exception in instruction generation: ", e)
                if num_tries >= num_retries:
                    return None
                num_tries += 1

    # save for debug
    if save_path:
        image = images[0] if isinstance(images, list) else images
        save_file_name = f"{save_path.stem}_{instruction}"[:MAX_FILE_NAME_CHAR]
        save_path = save_path.parent / f"{save_file_name}.jpg"
        visualize_step(image, actions, instruction, reasoning, save_path)

    return dict(reasoning=reasoning, instruction=instruction)
