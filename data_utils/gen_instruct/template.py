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

INTRO_OBS_ONLY = (
    "Imagine you are providing natural language instructions to guide a robot in an indoor environment. "
    "You will be given an image showing the current view, and a list of 8 of locations representing the "
    "movement you want the robot to execute. Each location corresponds to a point in [x, y] coordinate "
    "system within the current view. \n "
    "The x axis represents the vertical direction: positive values indicate distance in meters ahead, "
    "while negative values indicate distance behind. The y-axis represents the horizontal direction: "
    "positive values indicate distance to the left, and negative values indicate distance to the right.\n"
    "To answer the quesion: First, determine the general direction of the actions (e.g., forward, left, "
    "right, or backward). Next, identify where the end position aligns within the current view to "
    "understand where the movement is headed. Finally, decide on the best way to describe the direction. "
    "IMPORTANT:\n"
    "1.	Be succinct and answer questions directly with one of the given options.\n"
    "2.	Ensure your instruction is precise and unambiguous. When you refer to any object in sight, make "
    "sure only one object matches your description. Do not halluciate nor make up objects that are not"
    "in your current view.\n"
    "3.	Do not confuse doors with hallways. A Hallway has no barrier to open or close, and the passage is "
    "long, extending deeper into another area. A door would have a frame and a handle you could physically"
    "interact with. Specifically, you can enter a door into a room when you are in the hallway or exit a "
    "door when you are inside a room. \n"
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
    "move forward",
    "move towards {describe a goal in sight}",
    "take a {left or right} turn",
    "go around and backwards",
    "go along {describe the wall or corridor}",
    # "follow {a path or trajectory}",
    "go around {describe a obstacle or object to avoid}",
    "{enter or exit} {a door}",
]

INSTRUCT_TEMPLATES = [
    FREE_FORM,
    MAIN_DIRECT_4,
    MAIN_DIRECT_8,
    FORMAT_ACTION,
]
