from ..resources.action_designators import *
# from src.langchain.create_agents import *
from ..llm_configuration import *
from ..resources.failures import *
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Union
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from .global_custom_state import *
from ..resources.prompts.template_prompts import *
from .input_parser import *
from .instruct_agent import *

import re

ad_memory = MemorySaver()

action_classes = [PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction]

failure_reasons = [ObjectNotGraspedError,ObjectStillInContact,ObjectNotPlacedAtTargetLocation]

Concepts = [
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

class FailureSolution(BaseModel):
    """
    Inferred reasons for failure and suggested solution
    """
    failure_reasons : List[str] = Field(description="List of failure reasons inferred from the context")
    solution : List[str] = Field(description="The probable solution correct the failure")


class ParameterReasoner(BaseModel):
    """
    Inferred parameter-value updates needed and reasons for choosing them
    """
    updated_parameter_value : List[Dict[str,str]] = Field(description="List of updated parameter-value pairs")
    reason_parameter_value : List[str] = Field(description="List of reasons on why a value is chosen for the parameter")

# Agent State ---------------------------------------------------------------------------------------------------

action_designator_type = Union[PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction, str]

failure_reason_type = Union[ObjectNotGraspedError,ObjectStillInContact,ObjectNotPlacedAtTargetLocation, str]

class CustomStateInternal(AgentState):
    action_designator : action_designator_type
    reason_for_failure : failure_reason_type
    human_comment : str
    parameters_to_update: str
    failure_reasons_solutions: str
    updated_parameters : str
    update_parameters_reasons : str
    updated_action_designator: action_designator_type
    ad_instruction : str
    ad_human_instruction : str

def failure_reasoner_node(state: CustomStateInternal):
    print("INSIDE ANALYZER NODE")

    # Initialize variables
    structured_ollama = None
    failure_reasoner_prompt = ChatPromptTemplate.from_template(failure_reasoner_prompt_template_gemini)
    original_action_designator = ""
    update_reasons = ""

    # Extract inputs from state
    action_designator1 = state['action_designator']
    reason_for_failure1 = state['reason_for_failure']
    human_comment1 = state['human_comment']

    ad_human_instruction = ""
    if str(action_designator1) != "":
        ad_human_instruction = instructor_node(str(action_designator1))


    # # --- Parse failure reason ---
    # try:
    #     if isinstance(reason_for_failure1, str):
    #         failure_instance = eval(reason_for_failure1)  # Convert string to object
    #         failure_type = failure_instance.failure_type
    #
    #         # Identify matching failure class
    #         failure_cls = next(
    #             (cls for cls in failure_reasons if
    #              cls.__name__ == failure_type or getattr(cls, 'failure_type', None) == failure_type),
    #             None
    #         )
    #
    #         if failure_cls is None:
    #             raise ValueError(f"Unknown failure_type: {failure_type}")
    #
    #         # print("Failure Class:", failure_cls, type(failure_cls))
    #         error_message = failure_instance.args[0]
    #         # print(f"Error Message: {error_message}")
    # except Exception as e:
    #     print("Invalid failure type")

    # # --- Parse action designator ---
    # try:
    #     if isinstance(action_designator1, str):
    #         ad = eval(action_designator1)  # Convert string to object
    #         original_action_designator = str(ad)
    #         action_type = ad.action_type
    #
    #         # Identify matching action class
    #         action_cls = next(
    #             (cls for cls in action_classes if
    #              cls.__name__ == action_type or getattr(cls, 'action_type', None) == action_type),
    #             None
    #         )
    #
    #         if action_cls is None:
    #             raise ValueError(f"Unknown action_type: {action_type}")
    #
    #         # print("Original Action Designator:", ad)
    #         # print("Action Class:", action_cls, type(action_cls))
    #
    #         # Placeholder: structured_ollama output (currently not used)
    #         # structured_ollama = ollama_llm.with_structured_output(action_cls, method="json_schema")
    # except Exception as e:
    #     print("Unknown Action Designator")

    # --- Parsing ---
    failure_instance, error_message = parse_failure(reason_for_failure1)
    # ad_instance, action_cls = parse_designator(action_designator1)

    # original_action_designator = str(ad_instance)
    original_action_designator = state['action_designator']

    # structured_ollama = ollama_llm.with_structured_output(action_cls, method="json_schema")

    # --- Invoke analyzer chain ---
    chain = failure_reasoner_prompt | ollama_llm
    response = chain.invoke({
        "action_designator": original_action_designator,
        "reason_for_failure": reason_for_failure1 + f"Error Message: {error_message}",
        "human_comment": human_comment1
    })

    # --- Extract reasoning from <think> tags ---
    match = re.search(r"<think>(.*?)</think>", response.content, flags=re.DOTALL)
    if match:
        update_reasons_first = match.group(1).strip()
        structured_ollama = ollama_llm.with_structured_output(FailureSolution, method="json_schema")
        update_reasons = structured_ollama.invoke(input= (
        update_reasons_first +
        "\n\n Please analyze the reasoning above and respond in a conversational tone, as if you are explaining the issue and solution to a human." 
        "\nReturn the output as two lists:\n"
        "1. A list of precise, clearly worded reasons why the failure likely occurred.\n"
        "2. A list of practical and human-understandable suggestions that could help solve or avoid the failure." 
        "\nKeep the tone clear, natural, and helpful, while keeping each list item concise and informative." + " /nothink"))

    else:
        print("No <think> tags found.")

    # --- Clean response by removing <think> block ---
    if re.search(r"<think>.*?</think>", response.content, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
    else:
        cleaned_res = response.content.strip()

    # Final cleanup
    cleaned_res = cleaned_res.strip()

    # --- Return structured output ---
    return {
        "parameters_to_update": cleaned_res,
        "failure_reasons_solutions": update_reasons.model_dump_json(),
        "ad_human_instruction" : ad_human_instruction
    }

def context_facilitator_node(state: CustomStateInternal):
    print("INSIDE CONTEXT NODE")

    parameters_to_update1 = state["parameters_to_update"]
    update_reasons1 = state["failure_reasons_solutions"]
    human_comment1 = state["human_comment"]
    update_reasons = ""

    context_prompt = ChatPromptTemplate.from_template(context_prompt_template_gemini)

    chain = context_prompt | ollama_llm

    response = chain.invoke({"parameters_to_update" : parameters_to_update1,
                             "update_reasons" : update_reasons1,
                             "human_comment" : human_comment1,
                             "concepts" : Concepts})

    # --- Extract reasoning from <think> tags ---
    match = re.search(r"<think>(.*?)</think>", response.content, flags=re.DOTALL)
    if match:
        update_reasons_first = match.group(1).strip()
        structured_ollama = ollama_llm.with_structured_output(ParameterReasoner, method="json_schema")
        update_reasons = structured_ollama.invoke(
            input=(
                    update_reasons_first +
                    "\n\nPlease analyze the reasoning above and summarize it in a helpful, conversational tone — as if you're explaining your decisions to a human."
                    "\nReturn your response in two parts:\n"
                    "1. A list of updated parameter-value pairs that should be changed.\n"
                    "2. A list of clear, concise explanations for why each new value was chosen."
                    "\nEnsure the explanations are human-readable, natural in tone, and informative — but still concise and structured." + " /nothink"
            ))
    else:
        print("No <think> tags found.")

    # --- Clean response by removing <think> block ---
    if re.search(r"<think>.*?</think>", response.content, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
    else:
        cleaned_res = response.content.strip()

    return {"updated_parameters" : cleaned_res,
            "update_parameters_reasons" : update_reasons.model_dump_json()}

def updater_node(state: CustomStateInternal):
    print("INSIDE UPDATER NODE")

    # Initialize variables
    structured_ollama = None
    updater_prompt = ChatPromptTemplate.from_template(updater_prompt_template_gemini)
    clean_prompt = ChatPromptTemplate.from_template(clean_prompt_template)
    original_action_designator = ""

    # Extract inputs from state
    action_designator1 = state['action_designator']
    reason_for_failure1 = state['reason_for_failure']
    human_comment1 = state['human_comment']
    parameters_to_update1 = state['parameters_to_update']
    updated_parameters1 = state['updated_parameters']
    update_reasons1 = state['failure_reasons_solutions']
    update_parameters_reasons1 = state['update_parameters_reasons']

    # Debug prints
    print("Action Designator:", action_designator1)
    print("Reason for Failure:", reason_for_failure1)
    print("Human Comment:", human_comment1)

    # --- Parsing ---
    # failure_instance, error_message = parse_failure(reason_for_failure1)
    ad_instance, action_cls = parse_designator(action_designator1)

    original_action_designator = str(ad_instance)
    structured_ollama = ollama_llm.with_structured_output(action_cls, method="json_schema")

    chain = updater_prompt | structured_ollama

    # Final Output Shaper

    response = chain.invoke({
        "action_designator": action_designator1,
        "updated_parameters": updated_parameters1,
        "update_parameters_reasons": update_parameters_reasons1
    })

    print("Model Response:", response)

    # chain2 = clean_prompt | ollama_llm
    # cleaner_response = chain2.invoke({"class_ref" : str(state['action_designator']),
    #                "class_response" : response})

    # --- Return updated action designator ---
    return {
        "updated_action_designator": response
        # "updated_action_designator": cleaner_response.content
    }


#SoleGraphbuilder

graph_builder = StateGraph(CustomStateInternal)

graph_builder.add_node("failure_reasoner", failure_reasoner_node)
graph_builder.add_node("contexter", context_facilitator_node)
graph_builder.add_node("updater", updater_node)

graph_builder.set_entry_point("failure_reasoner")
graph_builder.add_edge("failure_reasoner", "contexter")
graph_builder.add_edge("contexter", "updater")
graph_builder.add_edge("updater", END)

sole = graph_builder.compile(checkpointer=ad_memory)


def designator_corrector_node(state : CustomState):

    failed_action_designator = state['action_designator']
    error = state.get("reason_for_failure","")
    human_comment = state.get("human_comment","")


    config = {"configurable": {"thread_id": 1}}

    result = sole.invoke({"action_designator": failed_action_designator, "reason_for_failure": error,
             "human_comment" : human_comment}, config = config, stream_mode="updates")

    update_parameters_reasons = sole.get_state(config).values["update_parameters_reasons"]
    updated_parameters = sole.get_state(config).values["updated_parameters"]
    updated_action_designator = sole.get_state(config).values["updated_action_designator"]
    failure_reasons_solutions = sole.get_state(config).values["failure_reasons_solutions"]
    parameters_to_update = sole.get_state(config).values["parameters_to_update"]
    ad_human_instruction = sole.get_state(config).values["ad_human_instruction"]

    return Command(
        update= {
            "parameters_to_update" : parameters_to_update,
            "failure_reasons_solutions" : failure_reasons_solutions,
            "update_parameters_reasons" : update_parameters_reasons,
            "updated_parameters" : updated_parameters,
            "updated_action_designator" : updated_action_designator,
            "human_instruction" : ad_human_instruction
        },
        goto=END
    )



if __name__ == "__main__":
    test_obj = Object(name="cup",concept="cup", color="blue")
    test_robot = Object(name="robot", concept="robot")
    test_links = [Link(name="gripper_link"), Link(name="wrist_link")]
    test_pose = PoseStamped(pose=Pose(
        position=Vector3(x=1.0, y=2.0, z=3.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)))

    action_designator = ("PickUpAction(object_designator=Object(name='cup',concept='cup', color='blue'), arm=Arms.LEFT, "
                         "grasp_description=GraspDescription(approach_direction=Grasp.TOP,vertical_alignment=Grasp.TOP, rotate_gripper=True))")
    grasping_error = ("ObjectNotGraspedErrorModel(obj=Object(name='cup',concept='cup', color='blue'), "
                      "robot=Object(name='robot', concept='robot'), arm=Arms.LEFT, grasp=Grasp.TOP)")
    human_comment = "pick up the yellow bottle not the blue cup"
    parameters_to_update = '{\n    "name": "cup",\n    "color": "blue"\n}',
    update_reasons = """
        "Okay, let's tackle this problem. The user wants me to analyze why the action failed and suggest parameter changes. \n\nFirst, looking at the action designator. The error is ObjectNotGraspedError, specifically mentioning a blue cup. The robot tried to grasp the cup with the left arm from the top. The user's comment says they wanted to pick up the yellow bottle, not the blue cup. \n\nSo the main issue here is that the action was trying to pick up the wrong object. The action designator probably had the object name as 'cup' which is blue. The user intended to pick up a yellow bottle. Therefore, the parameters related to the object name and color need to be changed. \n\nThe original action's object is 'cup' with color 'blue'. The correct object should be 'bottle' with color 'yellow'. So the parameters 'name' and 'color' in the action designator should be updated. The arm and grasp might still be correct, but since the object is wrong, those parameters aren't the issue here. \n\nTherefore, the parameters to modify are the object's name and color. The dictionary should include these changes."
     """

    config = {"configurable" : {"thread_id" : 1}}
    # for s in sole.stream({"action_designator": HumanMessage(content=action_designator), "reason_for_failure": HumanMessage(content=grasping_error),
    #              "human_comment": HumanMessage(content=human_comment)}, config = config):
    #     print(s)
    #     print("--------------")
    #
    #
    # print(sole.get_state(config))

    # sole.invoke({"action_designator": HumanMessage(content=action_designator), "reason_for_failure": HumanMessage(content=grasping_error),
    #              "human_comment": HumanMessage(content=human_comment)}, config = config)