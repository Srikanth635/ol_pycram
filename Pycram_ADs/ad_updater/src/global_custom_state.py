from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Union
from ..resources.action_designators import *
from ..resources.failures import *

action_designator_type = Union[PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction, str]

failure_reason_type = Union[ObjectNotGraspedError,ObjectStillInContact,ObjectNotPlacedAtTargetLocation, str]

class CustomState(AgentState):
    instruction: str
    action_designator : action_designator_type
    reason_for_failure : failure_reason_type
    human_comment: str
    parameters_to_update: str
    failure_reasons_solutions: str
    updated_parameters : str
    update_parameters_reasons : str
    updated_action_designator: action_designator_type
    human_instruction : str