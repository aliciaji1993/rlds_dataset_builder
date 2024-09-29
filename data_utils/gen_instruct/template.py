INTRODUCTION = (
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

FREE_FORM_INSTRUCTIONS = [
    "Describe the trajectory using natural language. Example instructions:",
    "Example 1. make a sharp right turn to turn away from the wall.",
    "Example 2. Turn slightly to the right, continue curving to the right into the hallway.",
    "Example 3. Move straight ahead towards the door in the right corner.",
    "Example 4. Turn slightly to the right to align with the hallway, then continue straight ahead.",
    "Example 5. Move a few steps forward, then curve to the right.",
]

MAIN_DIRECTIONS_4 = [
    "Pick the instruction that best describes the direction you want the robot to take:",
    "take a left turn",
    "take a right turn",
    "move forward",
    "move backward",
]

MAIN_DIRECTIONS_8 = [
    "Pick the instruction that best describes the direction you want the robot to take:",
    "turn left",
    "turn right",
    "move forward",
    "move backward",
    "move forward-left",
    "move forward-right",
    "move backward-left",
    "move backward-right",
]

FORMATTED_ACTIONS = [
    "Pick the instruction that best describes the direction you want the robot to take, replacing the brackets:",
    "move forward",
    "move towards {describe a goal in sight}",
    "go along {wall or corridor}",
    "go around {obstacle or object to avoid}",
    "go through {door or door frame}",
    "turn {left or right}",
    "turn around and backwards",
]

INSTRUCTION_TEMPLATES = [
    FREE_FORM_INSTRUCTIONS,
    MAIN_DIRECTIONS_4,
    MAIN_DIRECTIONS_8,
    FORMATTED_ACTIONS,
]
