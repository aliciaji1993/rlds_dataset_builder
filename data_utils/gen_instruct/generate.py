import base64
import itertools
import io
import numpy as np
import random

from enum import IntEnum

from openai import OpenAI
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

from .chat_wrapper import ChatWrapper


class InstructType(IntEnum):
    FREE_FORM_INSTRUCTIONS = 0
    MAIN_DIRECTIONS_4 = 1
    MAIN_DIRECTIONS_8 = 2
    FORMATTED_ACTIONS = 3


DOWN_SAMPLE_KEYWORDS = [
    "move forward",
]


def generate_instruction(
    chat: ChatWrapper,
    image: Image,
    actions: np.ndarray,
    instruction_prompt: str,
    save_path: Path,
    keyword_down_sample_rate: float = 0.05,
):
    fig, axs = plt.subplots(1, 2)

    # show observation image
    axs[0].imshow(image)
    axs[0].axis("off")
    axs[0].set_title(f"observation")

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

    # plot actions
    plot_actions(axs[1], actions)

    # draw fig on canvas and save to io buffer
    fig.tight_layout()
    # fig.canvas.draw()
    # fig_image = Image.frombytes(
    #     "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    # )

    # generate instruction from gpt-4o
    user_prompt = f"Given list of actions: {actions}\n" + instruction_prompt
    instruction = chat.send_message(image, user_prompt)
    axs[1].set_title(instruction)
    fig.suptitle(str(save_path.stem), fontsize=12)

    # save and close
    # if not (
    #     any([keyword in instruction for keyword in DOWN_SAMPLE_KEYWORDS])
    #     and random.random() > keyword_down_sample_rate
    # ):
    fig.savefig(save_path, dpi=300)
    plt.close()
