from typing import Any, Dict
import numpy as np
from PIL import Image


################################################################################################
#                                        Target config                                         #
################################################################################################
# features=tfds.features.FeaturesDict({
#     'steps': tfds.features.Dataset({
#         'observation': tfds.features.FeaturesDict({
#             'image': tfds.features.Image(
#                 shape=(128, 128, 3),
#                 dtype=np.uint8,
#                 encoding_format='jpeg',
#                 doc='Main camera RGB observation.',
#             ),
#             'state': tfds.features.Tensor(
#                 shape=(2,),
#                 dtype=np.float32,
#                 doc='Robot current location state, in (x, y) coordinates',
#             ),
#         }),
#         'action': tfds.features.Tensor(
#             shape=(2,),
#             dtype=np.float32,
#             doc='Robot movement action, in (x, y) coordinates",
#         ),
#         'discount': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Discount if provided, default to 1.'
#         ),
#         'reward': tfds.features.Scalar(
#             dtype=np.float32,
#             doc='Reward if provided, 1 on final step for demos.'
#         ),
#         'is_first': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on first step of the episode.'
#         ),
#         'is_last': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode.'
#         ),
#         'is_terminal': tfds.features.Scalar(
#             dtype=np.bool_,
#             doc='True on last step of the episode if it is a terminal step, True for demos.'
#         ),
#         'language_instruction': tfds.features.Text(
#             doc='Language Instruction.'
#         ),
#         'language_embedding': tfds.features.Tensor(
#             shape=(512,),
#             dtype=np.float32,
#             doc='Kona language embedding. '
#                 'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
#         ),
#     })
################################################################################################
#                                                                                              #
################################################################################################


def transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Maps step from source dataset to target dataset config.
    Input is dict of numpy arrays."""
    img = Image.fromarray(step["observation"]["image"]).resize(
        (128, 128), Image.Resampling.LANCZOS
    )
    state = step["observation"]["state"]
    action = step["action"]
    transformed_step = {
        "observation": {"image": np.array(img), "state": np.array(state)},
        "action": np.array(action),
    }

    # copy over all other fields unchanged
    for copy_key in [
        "discount",
        "reward",
        "is_first",
        "is_last",
        "is_terminal",
        "language_instruction",
        "language_embedding",
    ]:
        transformed_step[copy_key] = step[copy_key]

    return transformed_step
