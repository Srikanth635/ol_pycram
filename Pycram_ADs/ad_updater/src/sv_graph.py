from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from .supervisor import *
from typing import Union
from .graph import designator_corrector_node
from ..llm_configuration import *
from ..resources.action_designators import *
from .pycram_agent import *
from ..resources.failures import *
from .global_custom_state import *

action_designator_type = Union[PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction, str]

failure_reason_type = Union[ObjectNotGraspedError,ObjectStillInContact,ObjectNotPlacedAtTargetLocation, str]

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

memory = MemorySaver()


builder = StateGraph(CustomState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("designator_corrector_node", designator_corrector_node)
builder.add_node("pycram_node", pycram_node)
# builder.add_node("instructor_node", instructor_node)


builder.add_edge("designator_corrector_node", END)
builder.add_edge("pycram_node", END)
# builder.add_edge("instructor_node", END)
# # builder.add_node("web_researcher", web_research_node)
# builder.add_node("framenet", framenet_node)
# builder.add_node("flanagan", flanagan_node)
# graph = builder.compile()
sv_grapher = builder.compile(checkpointer=memory)
