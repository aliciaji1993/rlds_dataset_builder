import numpy as np

import matplotlib.pyplot as plt


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
