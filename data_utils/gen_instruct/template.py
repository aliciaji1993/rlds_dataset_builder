INTRO_OBS_AND_ACTION_MAP = (
    "Imagine you are providing natural language instructions to guide a robot in an indoor environment. "
    "On the left side, you will see an image showing the robot’s current view, and on the right side, "
    "an image of a 2D map representing the movement you want the robot to perform, starting at the "
    "green circle in the center and ending at the red circle on the edge.\n"
    "The positive x-axis indicates movement to the right, and the negative x-axis indicates movement to "
    "the left. The positive y-axis represents forward movement, while the negative y-axis indicates "
    "backward movement.\n"
    "IMPORTANT:\n"
    "1.	Start from the correct point—the green circle at the center of the map. \n"
    "2.	Ensure your instruction is concise and unambiguous, with only one object or goal that matches "
    "your description. Do not halluciate nor make up objects in your sight.\n"
    "3.	If multiple valid directions are visible, clarify your instruction to distinguish the intended "
    "direction from others. \n"
)

INTRO_OBS_AND_ACTION_STRING = (
    "Imagine you are providing natural language instructions to guide a robot in an indoor environment. "
    "You will be given an image showing the current view, and a list of 8 of locations representing the "
    "movement you want the robot to execute. Each location corresponds to a point in [x, y] coordinate "
    "system within the current view. \n"
    "The x axis represents the vertical direction: positive values indicate distance in meters ahead, "
    "while negative values indicate distance behind. The y-axis represents the horizontal direction: "
    "positive values indicate distance to the left, and negative values indicate distance to the right.\n"
    "Think step by step: First, review the list of locations to identify the overall direction (e.g., "
    "forward, left, right, or backward). Then, check the coordinates of the final step to see where it "
    "aligns in the current view and understand the movement’s target. Double-check the target’s location "
    "in the view, its relative position (front, left, right), and confirm this matches the coordinates "
    "of the final step to avoid errors. Finally, determine the best way to describe the trajectory.\n"
    "IMPORTANT:\n"
    "1.	Be succinct and answer questions directly with one of the given options.\n"
    "2.	Ensure your instruction is precise and unambiguous. When you refer to any object in sight, make "
    "sure only one object matches your description. Do not halluciate nor make up objects that are not"
    "in your current view.\n"
    "3.	Do not confuse doors with hallways. A Hallway has no barrier to open or close, and the passage is "
    "long, extending deeper into another area. A door would have a frame and a handle you could physically "
    "interact with. Specifically, you can enter a door into a room when you are in the hallway or exit a "
    "door when you are inside a room. \n"
    "4.	Do not confuse left and right direction! In mose cases, y value of the last few steps determines the "
    "direction of moving. Positive y values indicate you are moving to the left (positive y = left), "
    "while negative y values indicate movements to the right (negative y = right).\n"
    "5.	Pay close attention to trajectories with increasingly negative x values, as these typically "
    "occur in backward turning movements. In such cases, the turning direction is determined by the "
    "initial steps rather than the final steps of the trajectory. \n"
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{think step in step}",\n'
    '    "instruction": "{describe how to execute this trajectory}",\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)

INTRO_8_OBS_AND_ACTION_STRING = (
    "Imagine you are providing natural language instructions to guide a robot in an indoor environment. "
    "You will be presented with a trajectory of 8 steps that represent the movement you want the robot "
    "to follow, consisting of 8 image observations and a list of locations corresponding to points in "
    "an [x, y] coordinate system within the current view."
    "The x axis represents the vertical direction: positive values indicate distance in meters ahead, "
    "while negative values indicate distance behind. The y-axis represents the horizontal direction: "
    "positive values indicate distance to the left, and negative values to the right.\n"
    "Finally, decide how best to describe the trajectory.\n"
    "Think step by step: First, review the list of locations to identify the overall direction (e.g., "
    "forward, left, right, or backward). Then, check the coordinates of the final step to see where it "
    "aligns in the current view and understand the movement’s target. Double-check the target’s location "
    "in the view, its relative position (front, left, right), and confirm this matches the coordinates "
    "of the final step to avoid errors. Finally, determine the best way to describe the trajectory.\n"
    "IMPORTANT:\n"
    "1.	Be succinct and answer questions in one phrase.\n"
    "2.	Ensure your instruction is precise and unambiguous. When you refer to any object in sight, make "
    "sure only one object matches your description. Do not halluciate nor make up objects that are not"
    "in your current view.\n"
    "3.	Do not confuse doors with hallways. A Hallway has no barrier to open or close, and the passage is "
    "long, extending deeper into another area. A door would have a frame and a handle you could physically "
    "interact with. Specifically, you can enter a door into a room when you are in the hallway or exit a "
    "door when you are inside a room. \n"
    "4.	Do not confuse left and right direction! In mose cases, y value of the last few steps determines the "
    "direction of moving. Positive y values indicate you are moving to the left (positive y = left), "
    "while negative y values indicate movements to the right (negative y = right).\n"
    "5. Pay close attention to trajectories with increasingly negative x values, as these typically "
    "occur in backward turning movements. In such cases, the turning direction is determined by the "
    "initial steps rather than the final steps of the trajectory. \n"
    "Response Format:\n"
    "{\n"
    '    "reasoning": "{think step in step}",\n'
    '    "instruction": "{describe how to execute this trajectory}",\n'
    "}\n"
    "Ensure the response can be parsed by Python json.loads. Do not wrap the json codes in JSON markers."
)

FREE_FORM = [
    "Describe the trajectory using natural language. Example instructions:",
    "Example 1. make a sharp right turn to turn away from the wall.",
    "Example 2. Turn slightly to the right, continue curving to the right into the hallway.",
    "Example 3. Move straight ahead towards the door in the right corner.",
    "Example 4. Turn slightly to the right to align with the hallway, then continue straight ahead.",
    "Example 5. Move a few steps forward, then curve to the right.",
]

MAIN_DIRECT_4 = [
    "take a left turn",
    "take a right turn",
    "move forward",
    "move backward",
]

MAIN_DIRECT_8 = [
    "turn left",
    "turn right",
    "move forward",
    "move backward",
    "move forward-left",
    "move forward-right",
    "move backward-left",
    "move backward-right",
]

FORMAT_ACTION = [
    "move {left, right, foward}",
    "move {left, right} towards {describe the target destination}",
    "take a {left, right} turn",
    "turn around and backwards",
    "go along {describe the wall or corridor}",
    "go around {describe the obstacle}",
    "{enter or exit} {describe the door to go through}",
]

INSTRUCT_TEMPLATES = [
    FREE_FORM,
    MAIN_DIRECT_4,
    MAIN_DIRECT_8,
    FORMAT_ACTION,
]

INTRO_TEMPLATES = [
    INTRO_OBS_AND_ACTION_MAP,
    INTRO_OBS_AND_ACTION_STRING,
    INTRO_8_OBS_AND_ACTION_STRING,
]
