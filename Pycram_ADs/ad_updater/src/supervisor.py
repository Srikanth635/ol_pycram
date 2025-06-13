from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import END
from langgraph.types import Command
from ..llm_configuration import *
from langchain_core.prompts import ChatPromptTemplate
from .global_custom_state import *
from ..llm_configuration import *

system_prompt_template = """
    You are a supervisor managing a workflow between the following worker nodes: designator_corrector_node and pycram_node.

    Given the input, your job is to analyze its structure and select the correct node to process the request. If the request has been 
    fulfilled or cannot be processed, respond with FINISH.
    
    ### Instructions: ###
    
    - Your primary goal is to distinguish between a correction task (for designator_corrector_node) and a generation task (for pycram_node).
    - Analyze the input provided. If it is structured data containing a failed action, route it to designator_corrector_node.
    - If the input is a simple natural language string containing a command, route it to pycram_node.
    - Choose ONLY the node that is explicitly designed for the input provided.
    - If the input does not fit the description for either node, respond with FINISH.
    
    ### Nodes: ###
    
    - designator_corrector_node: Handles the correction of a failed action designator.
    
        - Call this node when the input is structured data containing an action_designator, a failure_reason, and an optional human_comment.
        - Example Input:  "action_designator": "type": "pick-up", "object": "cup", "color": "blue", "failure_reason": "object-not-found", "human_comment": 
        "try the other blue cup"
        
    - pycram_node: Handles the generation of a new action designator from a human instruction.
    
        - Call this node when the input is a simple string of text that represents a command.
        - Example Input: "pick the yellow bottle"
        
    Now, perform the task only on the given inputs,
    
    instruction : {instruction} \n
    action_designator : {action_designator} \n
    failure_reason : {reason_for_failure} \n
    human_comment : {human_comment} \n
    
"""

system_prompt = ChatPromptTemplate.from_template(system_prompt_template)

system_prompt_template_2 = """
    You are a supervisor managing a workflow between the following worker nodes: designator_corrector_node and pycram_node.

    Given the input, your job is to analyze its structure and select the correct node to process the request. If the request has been 
    fulfilled or cannot be processed, respond with FINISH.

    ### Instructions: ###

    - Your primary goal is to distinguish between a correction task (for designator_corrector_node) and a generation task (for pycram_node).
    - Analyze the input provided. If it is structured data containing a failed action, route it to designator_corrector_node.
    - If the input is a simple natural language string containing a command, route it to pycram_node.
    - Choose ONLY the node that is explicitly designed for the input provided.
    - If the input does not fit the description for either node, respond with FINISH.

    ### Nodes: ###

    - designator_corrector_node: Handles the correction of a failed action designator.

        - Call this node when the input is structured data containing an action_designator, a failure_reason, and an optional human_comment.
        - Example Input:  "action_designator": "type": "pick-up", "object": "cup", "color": "blue", "failure_reason": "object-not-found", "human_comment": 
        "try the other blue cup"

    - pycram_node: Handles the generation of a new action designator from a human instruction.

        - Call this node when the input is a simple string of text that represents a command.
        - Example Input: "pick the yellow bottle"

    Now, perform the task only on the given inputs,

    human_comment : {instruction}

"""

system_prompt2 = ChatPromptTemplate.from_template(system_prompt_template_2)


# Define router type for structured output
class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["designator_corrector_node", "pycram_node" , "FINISH"]

def supervisor_node(state: CustomState) -> Command[Literal["designator_corrector_node", "pycram_node" ,"__end__"]]:
    instruction = state.get("instruction", "")
    action_designator = state.get("action_designator", "")
    reason_for_failure = state.get("reason_for_failure", "")
    human_comment = state.get("human_comment", "")

    response : Router = None

    if action_designator != "":
        chain = system_prompt | ollama_llm.with_structured_output(Router)
        response = chain.invoke({'instruction': action_designator, 'action_designator': action_designator,
                                 'reason_for_failure': reason_for_failure, "human_comment" : human_comment})
    else:
        chain = system_prompt2 | ollama_llm.with_structured_output(Router)
        response = chain.invoke({'instruction': instruction})

    goto = response["next"]
    print(f"Next Worker: {goto}")
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)