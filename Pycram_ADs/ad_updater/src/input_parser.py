from typing import Tuple, Union
from ..resources.action_designators import *

from ..resources.failures import *

parsed_ad_type = Union[PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
    MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
    OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
    SearchAction, FaceAtAction]

failure_class_type = Union[ObjectNotGraspedError,ObjectStillInContact,ObjectNotPlacedAtTargetLocation, None]
failure_reasons = [ObjectNotGraspedError,ObjectStillInContact,ObjectNotPlacedAtTargetLocation]
action_classes = [PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction]

def parse_designator(designator: str) -> Tuple[parsed_ad_type, type]:
    try:
        if isinstance(designator, str):
            ad = eval(designator)
            # print("eval designator arm: ", ad.arm)
            action_type = ad.action_type

            if not action_type:
                raise ValueError("Missing 'action_type' in designator")

            action_cls = next(
                (cls for cls in action_classes if cls.__name__ == action_type or
                 getattr(cls, 'action_type', None) == action_type),
                None
            )
            if action_cls is None:
                raise ValueError(f"Unknown action_type: {action_type}")
            # print("Original Action Designator:", ad)
            return ad, action_cls

        raise TypeError("Input Designator must be passed as a string")

    except Exception as e:
        print("Unknown Action Designator:", e)
        raise ValueError("Unknown Action Designator")


def parse_failure(failure: str) -> Tuple[failure_class_type, str]:
    try:
        if not isinstance(failure, str):
            raise TypeError("Input failure must be a string")

        # Try to evaluate as a structured failure object first
        try:
            failure_instance = eval(failure)

            failure_type = getattr(failure_instance, "failure_type", None)
            if not failure_type:
                raise ValueError("Missing 'failure_type' attribute in failure instance")

            failure_cls = next(
                (cls for cls in failure_reasons if
                 cls.__name__ == failure_type or getattr(cls, "failure_type", None) == failure_type),
                None
            )

            if failure_cls is None:
                raise ValueError(f"Unknown failure_type: {failure_type}")

            error_message = failure_instance.args[0] if failure_instance.args else ""
            return failure_instance, error_message

        except Exception:
            # If eval fails, treat input as plain string message
            return None, failure.strip()

    except Exception as e:
        print(f"Invalid failure input: {e}")
        raise ValueError("Invalid failure input")




    # try:
    #     if isinstance(failure, str):
    #         failure_instance = eval(failure)
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
    #         error_message = failure_instance.args[0]
    #
    #         return failure_instance, error_message
    #
    #     raise TypeError("Input Failure Type must be passed as a string")
    #
    # except Exception as e:
    #     print("Invalid failure type")
    #     raise ValueError("Invalid failure type")


def parse(designator: str, failure: str):

    parse_designator(designator)
    parse_failure(failure)



if __name__ == "__main__":
    test_obj = Object(name="cup", concept="cup", color="blue")
    test_robot = Object(name="robot", concept="robot")
    test_links = [Link(name="gripper_link"), Link(name="wrist_link")]
    test_pose = PoseStamped(pose=Pose(
        position=Vector3(x=1.0, y=2.0, z=3.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)))

    action_designator = (
        "PickUpAction(object_designator=Object(name='Cup',concept='Cup', color='blue'), arm=Arms.LEFT, "
        "grasp_description=GraspDescription(approach_direction=Grasp.TOP,vertical_alignment=Grasp.TOP, rotate_gripper=True))")

    place_designator = """PlaceAction(object_designator=Object(name='Cup',concept='Cup', color='blue'), target_location= PoseStamped(pose=Pose(position=Vector3(x=1.0, y=2.0, z=3.0),
    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))), arm=Arms.LEFT)"""
    grasping_error = ("ObjectNotGraspedError(obj=Object(name='cup',concept='Cup', color='blue'), "
                      "robot=Object(name='robot', concept='robot'), arm=Arms.LEFT, grasp=Grasp.TOP)")
    # human_comment = "pick up the yellow cup not the blue cup"

    grasping_error2 = "object cup was not grasped by LEFT arm using top grasp"

    print(parse_designator(action_designator))
    print(parse_failure(grasping_error2))
