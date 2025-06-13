from Pycram_ADs.ad_updater.llm_configuration import *
from langchain_core.prompts import ChatPromptTemplate

template = """
You're a intelligent and smart programming agent. You can understand the code.

Given the user class string input try to respond with the parameters of the class string.

For example: If input class string is "User(name='Alice', address=Address(city='London'))"
You should respond with the class parameters in JSON encoded string like "name"="Alice", "address"=  "city"="London"

IMPORTANT CONSIDERATIONS:
- Dont try to modify, update, change any of the parameters or their values.
- Output should be strictly restricted only to the JSON encoded string of the parameters.
- Dont hallucinate and Dont add any explanations or feedback to the response


Do the above mentioned task for the user input:

class string : {class_string}


/nothink
"""

# inputs = ("PickUpAction(object_designator=Object(name='Cup',concept='Cup', color='blue'), arm=Arms.LEFT, "
#                          "grasp_description=GraspDescription(approach_direction=Grasp.TOP,vertical_alignment=Grasp.TOP, rotate_gripper=True))")

place_designator = """PlaceAction(object_designator=Object(name='Cup',concept='Cup', color='blue'), target_location= PoseStamped(pose=Pose(position=Vector3(x=1.0, y=2.0, z=3.0),
    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))), arm=Arms.LEFT)"""

test = """Person(name="Srikanth", work=Company(company = "TCS", profession=Profession(designation="scientist", experience=100)"""

chat_template = ChatPromptTemplate.from_template(template)

chain = chat_template | ollama_llm

print(chain.invoke({"class_string": test}).content)