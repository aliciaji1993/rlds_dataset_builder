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
    FREE_FORM = 0
    MAIN_DIRECT_4 = 1
    MAIN_DIRECT_8 = 2
    FORMAT_ACTION = 3


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
    image: np.ndarray,
    actions: np.ndarray,
    instruction_prompt: str,
    save_path: Path,
    action_as_text: bool = True,
    keyword_down_sample_rate: float = 0.05,
):
    fig, axs = plt.subplots(1, 2)

    # show observation image
    axs[0].imshow(image)
    axs[0].axis("off")

    # plot actions
    plot_actions(axs[1], actions)
    plt.figtext(
        0.5,
        0.0,
        str(actions),
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # generate instruction from gpt-4o
    if action_as_text:
        # Feed action list as part of text prompt
        user_prompt = "\n".join(
            [
                f"Given list of actions: {actions}",
                instruction_prompt,
                "Pick the instruction that best describes the given actions, replacing the brackets.",
            ]
        )
    else:
        # draw fig (obs + action plot) on canvas and send as image to VLM
        fig.canvas.draw()
        image = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        # use instruction prompt directly
        user_prompt = instruction_prompt
    instruction = chat.send_message(image, user_prompt)
    fig.suptitle(instruction, fontsize=12)

    # set generated instruction as action plot title

    # save and close
    # if not (
    #     any([keyword in instruction for keyword in DOWN_SAMPLE_KEYWORDS])
    #     and random.random() > keyword_down_sample_rate
    # ):
    fig.tight_layout()
    plt.figtext(
        0.5,
        0.06,
        str(save_path.stem),
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    save_file_name = f"{save_path.stem}_{instruction}"[:MAX_FILE_NAME_CHAR]
    fig.savefig(save_path.parent / f"{save_file_name}.jpg", dpi=300)
    plt.close()
