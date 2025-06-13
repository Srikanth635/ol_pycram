

## 🚀 Docker Usage

Here's how to get the application running using Docker.

### 1. Build the Docker Image

Navigate to the root directory of the project (where the `Dockerfile` is located) and run the following command to build the Docker image.

```bash
docker build -t {image_name}:latest .
```

> `docker build`: The command to build an image from a Dockerfile.  
> `-t {image_name}:latest`: Tags the image with a name and a version. Replace `{image_name}` with your desired name (e.g., `ol_pycram`).  
> `.`: Specifies the build context (the current directory).

### 2. Run the Docker Container

Once the image is successfully built, you can run it as a container.

#### With GPU Support

If your application requires GPU access, use this command (Recommended):

```bash
docker run -d --gpus=all -p 8081:5001 --name {container_name} {image_name}
```

#### Without GPU Support

For applications that do not need GPU resources, use the standard run command:

```bash
docker run -d -p 8081:5001 --name {container_name} {image_name}
```

**Command Breakdown:**

- `docker run`: Creates and starts a new container from an image.
- `-d`: Runs the container in detached mode (in the background).
- `--gpus=all`: Provides the container with access to all available GPUs on the host machine. (Requires the NVIDIA Container Toolkit).
- `-p 8081:5001`: Maps port 8081 on the host to port 5001 inside the container. Access your application at http://localhost:8081.
- `--name {container_name}`: Assigns a memorable name to your container for easier management. Replace `{container_name}` with your preferred name (e.g., `cont_olpycram`).
- `{image_name}`: The name of the image you built earlier.

### Accessing the Container

To get an interactive shell inside your running container, use the `docker exec` command:

```bash
docker exec -it {container_name} bash
```

## 🧪 Testing the Application

You can test the running service by sending a POST request to the `/update` endpoint. The inputs should be structured as shown in `Pycram_ADs/tests/testing_api.py`.

Here is an example Python script using the `requests` library to test the endpoint.

```python
import requests
import json

# The URL of the running service
url = 'http://localhost:8081/update'

# Example 1: Data with action designator, failure reason, and human comment
action_designator_str = "PickUpAction(object_designator=Object(name='Cup',concept='Cup', color='blue'), arm=Arms.LEFT, grasp_description=GraspDescription(approach_direction=Grasp.TOP,vertical_alignment=Grasp.TOP, rotate_gripper=True))"
grasping_error_str = "ObjectNotGraspedErrorModel(obj=Object(name='cup',concept='Cup', color='blue'),robot=Object(name='robot', concept='robot'), arm=Arms.LEFT, grasp=Grasp.TOP)"
human_comment_str = "pick up the yellow cup not the blue cup"

# Example 1: Data for Action Designator Correction/ Action Designator to Instruction Generation
data = {
    'action_designator': action_designator_str,
    'reason_for_failure': grasping_error_str,
    'human_comment': human_comment_str
}

# Example 2: Data for Instruction to Action Designator Generation
data1 = {
    'instruction': "pick the cup from the table"
}

# Example 3: Empty data
data_empty = {}

# --- Sending the requests ---

# Send the first data payload
print("--- Testing with full data payload ---")
try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an exception for bad status codes
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

print("\n" + "="*40 + "\n")

# Send the second data payload
print("--- Testing with instruction only ---")
try:
    response1 = requests.post(url, json=data1)
    response1.raise_for_status()
    print("Status Code:", response1.status_code)
    print("Response JSON:", response1.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

print("\n" + "="*40 + "\n")

# Send the empty data payload
print("--- Testing with empty data ---")
try:
    response_empty = requests.post(url, json=data_empty)
    response_empty.raise_for_status()
    print("Status Code:", response_empty.status_code)
    print("Response JSON:", response_empty.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```
