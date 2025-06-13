from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Literal, Union, Annotated, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langgraph.types import Command
from langgraph.graph import StateGraph, END
from ..llm_configuration import *
from langgraph.checkpoint.memory import MemorySaver
from ..llm_configuration import *
from langgraph.prebuilt.chat_agent_executor import AgentState
from .global_custom_state import *
import ast
from ..resources.action_designators import *
from ..resources.failures import *

pycram_memory = MemorySaver()


model_selector_prompt_template = """
    You are a precise robotic action model classifier. Your task is to analyze a user's instruction and select the minimal and most 
    appropriate list of action model(s) required to accomplish the task.

    ### Available Action Models ###
    - PickUpAction: Grasps and lifts an object. This is a complete action that includes reaching, gripping, and lifting.
    - PlaceAction: Sets an object down at a target location.
    - NavigateAction - Navigates the Robot to a target position.
    - SetGripperAction: Sets the gripper to a specific state, such as open or closed. Use this only when the instruction is explicitly about the gripper itself, not as part of picking or placing.
    - MoveTorsoAction: Moves the robot's torso vertically (up or down).
    - GripAction - Grip an object with the robot.
    - MoveAndPickUpAction - Navigate to `standing_position`, then turn towards the object and pick it up.
    - MoveAndPlaceAction - Navigate to `standing_position`, then turn towards the object and pick it up.
    - OpenAction: Opens a container, such as a drawer, cabinet, or box.
    - CloseAction: Closes a container, such as a drawer, cabinet, or box.
    - TransportAction: A composite action to move a specified object from its current location to a new destination. This implies the robot is already holding the object.
    - ReachToPickUpAction - Let the robot reach a specific pose before picking up an object.
    
    ### Guiding Principles ###
    - Be Minimalist:
        Only select the actions that are strictly necessary to follow the instruction.
    - Prefer Simple Actions:
        Favor combining simple, fundamental actions (e.g., PickUpAction). Only use a composite action like TransportAction if the instruction implies a single, continuous 
        process of moving an already-held object.
    - Exact Names:
        You must use the exact model names provided in the list above.
    - No Redundancy:
        Do not select multiple actions if one action covers the intent. For example, do not choose SetGripperAction if PickUpAction is already selected, as picking implies a grip.
    - Empty List for Irrelevance:
        If the instruction does not correspond to any available robotic action, return an empty list ([]).
    
    ### Output Format ###
    Your response must be only a list of string(s), with each string being a model name from the list above.
    
    Examples
    Instruction:
    "Go to the kitchen counter and get me the apple."
    Output:
    ["NavigateAction", "PickUpAction"]
    
    Instruction:
    "Open the top drawer."
    Output:
    ["OpenAction"]
    
    Instruction:
    "Take this bottle from me and put it on the table."
    Output:
    ["PickUpAction", "PlaceAction"]
    
    Instruction:
    "Place the bowl on the table"
    Output:
    ["PlaceAction"]
    
    Instruction:
    "Thanks, that's all for now."
    Output:
    []
    
    ---
    
    Now, for the given natural language instruction {input_instruction}, generate the output.

"""

model_populator_prompt_template = """
    You are a precise AI assistant that functions as a JSON generator for a robotics control system.

    Your goal is to populate a list of JSON objects based on action model schemas and a natural language instruction. You must 
    translate abstract actions into concrete, parameter-filled instances.
    
    Task
    You will be given three inputs:
    
    - instruction: A natural language command for the robot.
    - selected_models: A list of action model names that have been chosen to fulfill the instruction.
    - model_schemas: The JSON schemas corresponding to each of the selected_models.

    Your task is to generate a valid JSON array where each object is an instantiated version of the corresponding action model from the selected_models list.

    Core Principles
    - Schema is Truth: The provided model_schemas are your single source of truth for structure, field names, data types, and which fields are required.
    - Mandatory Fields First: Identify the required fields in each schema. You must extract a value for these fields from the instruction. If a value for a
        required field cannot be reasonably extracted from the instruction, make it as unknown.
    - High-Confidence Optional Fields: Only populate optional fields (those not in the required list) if the instruction provides an explicit and
        unambiguous value for them. If you are not 100% sure, do not include the field. It is better to omit an optional field than to guess its value.
    - No Placeholders: Do not invent values or use generic placeholders for missing information.
    
    **SPECIAL GUIDANCE for Object Designator Parameters (`concept` and `name`):**

    * The `concept` parameter specifies the ontological category of an object, and its value **MUST** be chosen from the provided `concepts` list.
    * The `name` parameter typically describes a specific instance of an object (e.g., 'red cup', 'blue box').
    * **Interdependency:** `concept` and `name` are often linked. If one is being updated, consider if the other needs a corresponding adjustment to maintain consistency.
        * *Example:* If `name` was 'mug' and `concept` was 'Plate', and the reason suggests it should be a 'cup', then update `concept = 'Cup'` and potentially `name = 'cup'` (or keep original name if context allows).
        * **Name Changes:** Only change the `name` parameter if it is explicitly suggested by the `human_comment` or if it is clearly inconsistent with a newly selected `concept` or the `update_reasons`. Otherwise, keep the original `name` or refine it minimally. Ignore case sensitivity when matching names (e.g., 'red cup' should match 'Cup' concept).
        
    * Concept value can be only from this allowed list. It should be chosen from this exhaustive list and relevant to the name.
    concepts = [
        "World", "Floor", "Milk", "Robot", "Cereal", "Kitchen", "Food", "Fruit", "Apple",
        "Environment", "Apartment", "Cup", "Spoon", "Bowl", "PreferredGraspAlignment",
        "XAxis", "YAxis", "NoAlignment", "Truthy", "Falsy", "Cabinet", "Washer",
        "Drawer", "Refrigerator", "Sink", "Door", "Cutting", "Pouring", "Handle", "Link",
        "PhysicalObject", "PouringTool", "CuttingTool", "MixingTool", "Agent", "Human",
        "Room", "Location", "Container", "Joint", "ContinuousJoint", "HingeJoint",
        "FixedJoint", "MovableJoint", "FloatingJoint", "PlanarJoint", "PrismaticJoint",
        "RevoluteJoint", "DesignedFurniture", "Surface", "PhysicalTask", "Action", "Event",
        "Entity", "Task", "RootLink", "Supporter", "SupportedObject"
    ]
            
    ### Output Format ###
    - Your entire output must be similar to the provided model schemas with updated or inferred values.
    - Do not include any conversational text, explanations, or comments in your response.
    - If you cannot populate a required field for an action, you can return an error object for that specific action within the JSON array.
    
    ---
    
    Now, generate the structured action model instances for the following:
    instruction: {instruction}
    selected_models: {selected_models}
    model_schemas: {model_schemas}

"""


class ActionNames(BaseModel):
    model_names: List[Literal["PickUpAction", "PlaceAction", "NavigateAction", "SetGripperAction",
    "MoveTorsoAction", "GripAction", "MoveAndPickUpAction", "MoveAndPlaceAction", "OpenAction",
    "CloseAction", "ReachToPickUpAction", "TransportAction"]] = Field(description="Action model names")

# ActionsTypes = Annotated[
#     Union[
#         MoveTorsoAction, SetGripperAction, GripAction, ParkArmsAction, NavigateAction,
#         PickUpAction, PlaceAction, ReachToPickUpAction, TransportAction, LookAtAction,
#         OpenAction, CloseAction, FaceAtAction, DetectAction, SearchAction,
#         GraspingAction, MoveAndPickUpAction, MoveAndPlaceAction
#     ],
#     Field(discriminator="action_type")
# ]
#
# class Actions(BaseModel):
#     models : List[ActionsTypes] = Field(description="list of instantiated action model instances")

action_classes = [PickUpAction, NavigateAction, PlaceAction, SetGripperAction,
                  MoveTorsoAction, GripAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction]

action_classes_maps = {
    "PickUpAction": PickUpAction,
    "NavigateAction": NavigateAction,
    "PlaceAction": PlaceAction,
    "SetGripperAction": SetGripperAction,
    "MoveTorsoAction": MoveTorsoAction,
    "GripAction": GripAction,
    "ParkArmsAction": ParkArmsAction,
    "MoveAndPickUpAction": MoveAndPickUpAction,
    "MoveAndPlaceAction": MoveAndPlaceAction,
    "OpenAction": OpenAction,
    "CloseAction": CloseAction,
    "GraspingAction": GraspingAction,
    "ReachToPickUpAction": ReachToPickUpAction,
    "TransportAction": TransportAction,
}

class Actions(BaseModel):
    models : List[Union[*action_classes]] = Field(description="list of instantiated action model instances")

class PyCRAMState(TypedDict):
    action_names : Annotated[list, add_messages]
    action_models : Annotated[list, add_messages]


model_selector_prompt = ChatPromptTemplate.from_template(model_selector_prompt_template)
model_populator_prompt = ChatPromptTemplate.from_template(model_populator_prompt_template)

structured_ollama_llm_pc1 = ollama_llm.with_structured_output(ActionNames, method="json_schema")

structured_ollama_llm_pc2 = ollama_llm.with_structured_output(Actions, method="json_schema")

#
# @tool(description="PyCram Action Designator pydantic model selector tool",
#       return_direct=True,)
# def model_selector(instruction : str):
#     """
#     PyCram Action Designator model selector tool that selects relevant Pydantic model names
#     based on the input robot task instruction.
#
#     :param instruction: Natural language instruction describing the robot task.
#     :return: List of relevant action model class names (as strings).
#     """
#     print("INSIDE MODEL SELECTOR TOOL")
#     print("The instruction is :", instruction)
#     answers["instruction"] = instruction
#     chain = model_selector_prompt | structured_ollama_llm_pc1
#     response = chain.invoke({"input_instruction": instruction})
#     # json_response = response.model_dump_json(indent=2, by_alias=True)
#     response_python_dict = response.model_dump()
#     answers["model_names"] = response_python_dict["model_names"]
#     print("response of tool 1 : ", type(response), response)
#     # framenet_answers.append(json_response)
#     return response_python_dict
#
#
# class Populater(BaseModel):
#     instruction : str = Field()
#     model_names : List[str] = Field()
#
# @tool(description="PyCram Action Designator model populator tool that populates Pydantic models",
#       return_direct=True, args_schema=Populater)
# def model_populator(instruction : str , model_names : List[str]) -> dict :
#     """
#     PyCram Action Designator model populator tool that populates Pydantic models
#
#     :param instruction: Natural language instruction describing the robot task.
#     :param model_names: list of selected pydantic action model class names (as strings) for the robot task.
#     :return: dictionary
#     """
#     print("INSIDE MODEL POPULATOR TOOL")
#     instruction_for_populator = {
#         "instruction": instruction,
#         "selected_models": model_names
#     }
#     chain = model_populator_prompt | structured_ollama_llm_pc2
#     response = chain.invoke(instruction_for_populator)
#     # return {"populated_models" : response.models}
#     return response.model_dump()

# model_selector_tool_direct_return = Tool.from_function(
#     func=model_selector,
#     name= "model_selector",
#     description= "PyCram Action Designator pydantic model selector tool",
#     return_direct=True
# )

# model_populator_tool_direct_return = Tool.from_function(
#     func=model_populator,
#     name= "model_populator",
#     description= "PyCram Action Designator pydantic model populator tool",
#     return_direct=True  # ✅ This ensures the agent returns it as-is
# )

# model_populator_tool_direct_return = StructuredTool(func=model_populator,
#                name= "model_populator",
#                description= "PyCram Action Designator pydantic model populator",
#                args_schema=Populater,
#                return_direct=True)

# model_populator_tool_direct_return = StructuredTool.from_function(
#     func=model_populator,
#     name= "model_populator",
#     description= "PyCram Action Designator pydantic model populator tool",
#     return_direct=True,
#     args_schema=Populater
# )

# Agent Specific System Prompt
sys_prompt_content = """
    You are a robotic action planning agent that helps convert user instructions into structured robot actions.

    You have access to two tools:

    ---

    ### 🔧 TOOL 1: `model_selector(instruction: str) -> List[str]`

    - Takes a natural language instruction from the user (e.g., "Pick up the red cup and place it on the table")
    - Returns a list of valid action model names, such as:
      ["PickUpActionModel", "PlaceActionModel"]
    - These names correspond to specific Pydantic models for robotic actions
    - Output is strictly limited to known model names

    Use this tool first to decide which action(s) are relevant.

    ---

    ### 🧩 TOOL 2: `model_populator(instruction : str, model_names : List[str] ) -> dict`

    - Takes the user input NL instruction of robot task and a list of selected model names from Tool 1
    - Returns structured and populated Pydantic model instances
    - Each instance will contain all required parameters for execution (e.g., object name, pose, arm, etc.)
    - If something is missing from the original user instruction, use defaults or make reasonable assumptions
    - Do not invent model names not listed in the tool input

    ---

    ### 📝 Goals:

    - Use Tool 1 to **determine** what types of robot actions are needed.
    - Use Tool 2 to **generate structured input** for each action model returned from Tool 1.
    - Your job is to coordinate these tools to map freeform user instructions into a fully specified set of structured robot commands.

    ---

    ### 🧪 Example Workflow:

    #### User:  
    "Move to the table and pick up the bottle"

    #### Tool 1 Output:  
    ["NavigateAction", "PickUpAction"]

    #### Tool 2 Output:  
    [
      {
        "target_location": {
          "position": [1.0, 0.5, 0.0],
          "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        "keep_joint_states": true
      },
      {
        "object_designator": "bottle_1",
        "arm": "left",
        "grasp_description": "front_grasp"
      }
    ]
    
    Only return structured actions at the end of both steps. Follow this tool-based reasoning pipeline strictly.
    If the instruction is unclear, make assumptions but never skip steps.
"""

sys_prompt_content_short = """
    
    You are a robotic action planning agent converting user instructions into structured robot actions using two tools.
    
    ---
    
    ### 🔧 TOOL 1: `model_selector(instruction: str) -> List[str]`
    
    - Input: a natural language task (e.g., "Pick up the red cup and place it on the table")
    - Output: a list of valid model names like ["PickUpActionModel", "PlaceActionModel"]
    - These models represent predefined robotic actions
    - Only return known model names
    
    Always start with this tool to identify required actions.
    
    ---
    
    ### 🧩 TOOL 2: `model_populator(instruction: str, model_names: List[str]) -> dict`
    
    - Input: the same user instruction and the output from Tool 1
    - Output: populated instances of each action model (e.g., object, pose, arm)
    - Use defaults or reasonable assumptions when needed
    - Do not invent new model names
    
    ---
    
    ### 📦 CONTEXT: CRAM-style Action Designators
    
    You may be given **action designators** with detailed info about the object, location, tool, and action.
    
    These are read-only and help you:
    
    - Resolve ambiguities
    - Infer missing parameters
    - Improve reasoning about task context
    
    Never generate or modify them.
    
    ---
    
    ### 📝 Objective:
    
    - Use Tool 1 to select actions.
    - Use Tool 2 to populate structured inputs.
    - Use action designators as **context** only.
    - Produce fully structured robot commands from freeform input.
    
    ---
    
    ### 🧪 Example 1
    
    User: "Move to the table and pick up the bottle"
    
    Tool 1 → ["NavigateAction", "PickUpAction"]
    
    Tool 2 → [
      {
        "target_location": {
          "position": [1.0, 0.5, 0.0],
          "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        "keep_joint_states": true
      },
      {
        "object_designator": "bottle_1",
        "arm": "left",
        "grasp_description": "front_grasp"
      }
    ]
    
    ---
    
    ### 🧪 Example 2
    
    User: "Pick up the bottle near the sink"
    
    Provided Action Designator:
    { "action": {"type": "PickingUp"}, "object": { "name": "Bottle", "properties": {"material": "plastic", "color": "transparent"} }, "location": {"name": "SinkCounter"} }
    
    Tool 1 → ["PickUpAction"]
    
    Tool 2 → { "object_designator": "bottle_1", "arm": "right", "grasp_description": "top_grasp" }
    
    ---
    
    ### 🤖 REMINDERS
    
    - Always follow: Tool 1 → Tool 2
    - Use designators only for reference
    - Never skip tools, even if designators seem complete
    - Fill in missing details with commonsense assumptions
    
    Follow the pipeline strictly.
"""
pycram_agent_sys_prompt = SystemMessage(content=sys_prompt_content)

# # # Create the agent
# pycram_agent = create_agent(ollama_llm, [model_selector, model_populator],
#                             agent_sys_prompt=pycram_agent_sys_prompt)



class CustomStateInternal2(AgentState):
    action_designator : action_designator_type
    reason_for_failure : failure_reason_type
    instruction : str
    parameters_to_update: str
    failure_reasons_solutions: str
    updated_parameters : str
    update_parameters_reasons : str
    updated_action_designator: action_designator_type
    model_names : str
    pycram_model : str


# Nodes

def model_selector_node(state : CustomStateInternal2):
    """
    PyCram Action Designator model selector tool that selects relevant Pydantic model names
    based on the input robot task instruction.
    """
    print("INSIDE MODEL SELECTOR TOOL")
    instruction = state['instruction']
    print("The instruction is :", instruction)
    # answers["instruction"] = instruction

    chain = model_selector_prompt | structured_ollama_llm_pc1
    response = chain.invoke({"input_instruction": instruction})
    # json_response = response.model_dump_json(indent=2, by_alias=True)
    response_python_dict = response.model_dump()
    mod_names = response_python_dict["model_names"]
    print("response of tool 1 : ", type(response), response)
    # framenet_answers.append(json_response)
    return {'model_names' : str(mod_names)}

def model_populator_node(state : CustomStateInternal2):
    """
    PyCram Action Designator model populator tool that populates Pydantic models
    """
    print("INSIDE MODEL POPULATOR TOOL")

    instruction = state['instruction']
    model_names = state['model_names']
    # model_names = ["PickUpAction"]
    print("The instruction is :", instruction)
    print("Model Names", model_names)

    context_schema = ""
    try:
        # 1. Safely parse the string into a list of names. This is done ONCE.
        model_names_eval = ast.literal_eval(model_names)
        # 2. Loop through the list of model name strings
        for model_name in model_names_eval:
            # 3. Look up the corresponding class from your mapping dictionary
            ActionClass = action_classes_maps.get(model_name)
            # 4. Check if the class was found and create an instance
            if ActionClass:
                model_schema = ActionClass.model_json_schema()
                # 5. Add the newly created instance to your list
                context_schema = context_schema + "\n" + f'{model_name} : {model_schema}'
                print(f"Successfully created instance of: {model_name}")
            else:
                print(f"Warning: Model name '{model_name}' not found in AVAILABLE_ACTIONS.")
    except (ValueError, SyntaxError) as e:
        print(f"Error: Could not parse the input string. Details: {e}")


    # context_schema = ""
    # for mod_name in model_names:
    #     mod_ins_ref = action_classes_maps.get(mod_name, None)
    #     if mod_ins_ref is not None:
    #         mod_ins = mod_ins_ref()
    #     else:
    #         mod_ins = None
    #
    #     print("Model Instance", mod_ins)
    #     if mod_ins is not None:
    #         model_schema = mod_ins.model_json_schema()
    #         print("Model Schema", model_schema)
    #         context_schema = context_schema + "\n" + f'{mod_name} : {model_schema}'

    print("Context Schema", context_schema)

    chain = model_populator_prompt | ollama_llm.with_structured_output(Actions, method="json_schema")
    response = chain.invoke({"instruction" : instruction, "selected_models" : model_names, "model_schemas" : context_schema})
    response_python_dict = response.model_dump()
    print("response :", str(response))
    # mods = response_python_dict["models"]
    return {'pycram_model' : str(response)}


graph_builder = StateGraph(CustomStateInternal2)
graph_builder.add_node("model_selector_node", model_selector_node)
graph_builder.add_node("model_populator_node", model_populator_node)

graph_builder.set_entry_point("model_selector_node")

graph_builder.add_edge("model_selector_node", "model_populator_node")
graph_builder.add_edge("model_populator_node", END)

pysole = graph_builder.compile(checkpointer=pycram_memory)





# Agent as Node
def pycram_node(state: CustomState):

    instruction = state['instruction']

    result = pysole.invoke({'instruction' : instruction})

    return Command(
        update={
            "updated_action_designator": result['pycram_model']
        },
        goto=END,
    )

# Agent as Node
# def pycram_node_pal(state: MessagesState):
#     # messages = [
#     #                {"role": "system", "content": framenet_system_prompt},
#     #            ] + state["messages"]
#     result = pycram_agent.invoke(state)
#     # print("Pycram agent results: ", type(result),result)
#     return {
#             "messages": result["messages"][-1]
#         }


if __name__ == '__main__':
    print(Arms.LEFT)
    # print(model_descriptions)

    # responsed = pycram_agent.invoke({"messages" : [HumanMessage(content="generate pycram base models for the instruction pick up the mug from the table")]})
    # print(responsed)

    # chain = model_populator_prompt | structured_ollama_llm_pc2
    #
    # print(chain)
    # model_populator.invoke("pick up the cup from the table")