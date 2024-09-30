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


DOWN_SAMPLE_KEYWORDS = [
    "move forward",
]

MAX_FILE_NAME_CHAR = 255


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


def generate_instruction(
    chat: ChatWrapper,
    images: Union[List[np.ndarray], np.ndarray],
    actions: np.ndarray,
    instruction_prompt: str,
    context_type: ContextType,
    save_path: Path = None,
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
        # use instruction prompt directly
        user_prompt = instruction_prompt
        generated_text = chat.send_message(obs_and_action_map, user_prompt)
        plt.close()
    else:
        # Feed action list as part of text prompt
        user_prompt = "\n".join(
            [
                f"Given list of actions: {actions}",
                instruction_prompt,
                "Pick the instruction that best describes the given actions, replacing the brackets.",
            ]
        )
        generated_text = chat.send_message(images, user_prompt)

    reasoning = json.loads(generated_text)["reasoning"]
    instruction = json.loads(generated_text)["instruction"]

    # save for debug
    if save_path:
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(
            instruction,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            wrap=True,
        )
        # draw fig (obs + action plot) on canvas
        axs[0].imshow(images[0] if isinstance(images, list) else images)
        axs[0].axis("off")
        plot_actions(axs[1], actions)
        plt.figtext(
            0.0,
            0.0,
            reasoning,
            wrap=True,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=9,
        )
        fig.tight_layout()
        save_file_name = f"{save_path.stem}_{instruction}"[:MAX_FILE_NAME_CHAR]
        fig.savefig(save_path.parent / f"{save_file_name}.jpg", dpi=300)
        plt.close()
