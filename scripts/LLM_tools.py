import openai
import json
import numpy as np
import ast
import yaml


def describe_space(objects_dict):
    with open("/root/UAV-control-using-semantics-and-LLM/config/chatgpt_credentials.yaml", "r") as file:
        data = yaml.safe_load(file)

    openai.api_key = data['api_key']

    objects_dict = {k: [v2.tolist() for v2 in v] for k, v in objects_dict.items()}
    objects_json_str = json.dumps(objects_dict)
    messages = [
    {
        "role": "system",
        "content": (
            "I am mapping a space and want a short description of it when done. "
            "You are a helpful assistant that uses logic of objects around me and their position in space to create a description of the mapped space. "
            "You must always respond with valid JSON without Markdown containing exactly one key: "
            "'description' (a short string containing the description of the space in a few sentences). "
            "Do not include any text outside of the JSON. IMPORTANT - Without Markdown! Keep the text proffesional. "
            "Use objects in connection with other objects in its proximity to develop a more detailed description. "
        )
    },
    {
        "role": "user",
        "content": (
            f"Given the following dictionary of objects and their 3D coordinates: \n{objects_json_str}\n "
            "in valid JSON, return a short description of the mapped space. "
        )
    }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        temperature=0.4
    )
    assistant_reply = response.choices[0].message.content

    try:
        data = json.loads(assistant_reply)
        description = data["description"]
        return description 

    except json.JSONDecodeError:
        print("Failed to parse the assistant's reply as valid JSON.")
    except KeyError as e:
        print(f"Missing expected key in JSON response: {e}")

    
    

def define_target_object(input_description):
    with open("/root/UAV-control-using-semantics-and-LLM/config/chatgpt_credentials.yaml", "r") as file:
        data = yaml.safe_load(file)
    openai.api_key = data['api_key']

    messages = [
    {
        "role": "system",
        "content": (
            "I am searching for something. "
            "You are a helpful assistant that will parse my description of what i am searching for and give me some help in finding it. "
            "You must always respond with valid JSON containing exactly two keys: "
            "'target object' (the object from COCO dataset I am searching for), 'helper objects' (a list of few objects from MS COCO Dataset). "
            "Do not include any text outside of the JSON. IMPORTANT - Without Markdown!. "
            "Using the description i gave you, give me a list of objects near which i can find my object. "
            "The objects must only be from a MS COCO dataset and should be written all lower caps. "
        )
    },
    {
        "role": "user",
        "content": (
            f"Given the following description of what I am searching for: \n{input_description}\n "
            "in valid JSON, return a list of a few objects near which I can find my one. They must be one of 80 classes of MS COCO dataset. "
        )
    }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        temperature=0
    )
    assistant_reply = response.choices[0].message.content
    #print(assistant_reply)
    try:
        data = json.loads(assistant_reply)
        target_object = data["target object"]
        helper_objects = data["helper objects"]
        return target_object, helper_objects 

    except json.JSONDecodeError:
        print("Failed to parse the assistant's reply as valid JSON.")
    except KeyError as e:
        print(f"Missing expected key in JSON response: {e}")

def define_target_object1(input_description):
    with open("/root/UAV-control-using-semantics-and-LLM/config/chatgpt_credentials.yaml", "r") as file:
        data = yaml.safe_load(file)
    openai.api_key = data['api_key']

    messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant that will parse my description of what I am searching for and give me some help in finding it. "
            "You must always respond with valid JSON containing exactly two keys: "
            "'target object' (the object I am searching for), 'helper objects' (a list of few objects that are connected to the search querry). "
            "Do not include any text outside of the JSON. IMPORTANT - Without Markdown!. "
            "Using the description I gave you, define the target object class and produce a list of objects near which I can find my object - helper ones. "
            "Make the list contain 10 to 20 objects and they should be relevant to the target object and should also contain the target object."
            "Take into consideration co-occurrence, user descriptions and instructions. "
            "Be more specific and not too vauge or general. The objects should be eligible for a YOLOE object detector class. "
        )
    },
    {
        "role": "user",
        "content": (
            f"Given the following description of what I am searching for: \n{input_description}\n "
            "in valid JSON, define the target object and a list of a few objects near which I can find my one and are relevant to the search."
        )
    }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        temperature=0
    )
    assistant_reply = response.choices[0].message.content
    #print(assistant_reply)
    try:
        data = json.loads(assistant_reply)
        target_object = data["target object"]
        helper_objects = data["helper objects"]
        return target_object, helper_objects 

    except json.JSONDecodeError:
        print("Failed to parse the assistant's reply as valid JSON.")
    except KeyError as e:
        print(f"Missing expected key in JSON response: {e}")

    

def decide_movement1(objects_dict, target_description):
    with open("/root/UAV-control-using-semantics-and-LLM/config/chatgpt_credentials.yaml", "r") as file:
        data = yaml.safe_load(file)

    openai.api_key = data['api_key']

    objects_dict = {k: [v2.tolist() for v2 in v] for k, v in objects_dict.items()}
    objects_json_str = json.dumps(objects_dict)
    messages = [
    {
        "role": "system",
        "content": (
            f"I am searching for some object and I use YOLO NN to detect some objects from MS COCO Dataset around. "
            f"You are a helpful assistant that uses logic from the description I give you and objects around me to deduct where my searched object could be and why. "
            "You must always respond with valid JSON containing exactly five keys: "
            "'lost' (MS COCO dataset label of searched object), 'found' (string 'True' or 'False'), 'object' (a string), 'coordinates' (a list of x,y,z coordinates) and 'explanation' (string). "
            "Do not include any text outside of the JSON. Do not include any Markdown! "
            f"From the provided description and dictionary, understand what I am searching for and return the object that has the highest possibility of having that in its proximity. "
            f"BE STRICT! - only high connections with my object! - If none of the given objects are connected to the searched one, return 'False' in the 'found' field, and empty slots in other fields. "
            "If there are multiple objects with the same label, use other objects around them to deduct which has "
            f"the higher probability of containing my object close to it."
        )
    },
    {
        "role": "user",
        "content": (
            f"{target_description}"
            f"Given what I am searhing for and the following dictionary of objects with their 3D coordinates: \n{objects_json_str}\n "
            f"in valid JSON, return the object with the highest possibility of having the object i am searching for close to it. "
            "For objects in more than one place, give them additional meaning by connecting objects close to them. "
        )
    }
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        temperature=0
    )
    assistant_reply = response.choices[0].message.content

    try:
        print(assistant_reply)
        data = json.loads(assistant_reply)
        found_flag = data["found"]
        object_name = data["object"]
        coordinates = data["coordinates"]
        explanation = data["explanation"]
        return ast.literal_eval(found_flag), object_name, coordinates, explanation

    except json.JSONDecodeError:
        print("Failed to parse the assistant's reply as valid JSON.")
    except KeyError as e:
        print(f"Missing expected key in JSON response: {e}")


def decide_movement(objects_dict, target_description):
    from openai import OpenAI

    with open("/root/UAV-control-using-semantics-and-LLM/config/chatgpt_credentials.yaml", "r") as file:
        data = yaml.safe_load(file)

    client = OpenAI(api_key=data['api_key'])

    objects_dict = {k: [v2.tolist() for v2 in v] for k, v in objects_dict.items()}
    objects_json_str = json.dumps(objects_dict)

    prompt = (
            f"The USER is searching for some object and I use YOLO NN to detect some objects from MS COCO Dataset around. "
            f"You are a helpful assistant that uses logic from the description I give you and objects around me to deduct where my searched object could be and why. "
            "You must always respond with valid JSON containing exactly five keys: "
            "'lost' (MS COCO dataset label of searched object), 'found' (string 'True' or 'False'), 'object' (a string), 'coordinates' (a list of x,y,z coordinates) and 'explanation' (string). "
            "Do not include any text outside of the JSON. Do not include any Markdown! "
            f"From the provided description and dictionary, understand what the USER is searching for and return the object that has the highest possibility of having that in its proximity. "
            f"BE STRICT! - only high connections with target object! - If none of the given objects are connected to the searched one, return 'False' in the 'found' field, and empty slots in other fields. "
            "If there are multiple objects with the same label, use other objects around them to deduct which has "
            f"the higher probability of containing target object close to it."
            f"The USER gave instructions: {target_description}"
            f"Given what I am searhing for and the following dictionary of objects with their 3D coordinates: \n{objects_json_str}\n "
            f"in valid JSON, return the object with the highest possibility of having the object i am searching for close to it. "
            "For objects in more than one place, give them additional meaning by connecting objects close to them. "
    )

    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
    )


    assistant_reply = response.choices[0].message.content

    try:
        print(assistant_reply)
        data = json.loads(assistant_reply)
        found_flag = data["found"]
        object_name = data["object"]
        coordinates = data["coordinates"]
        explanation = data["explanation"]
        return ast.literal_eval(found_flag), object_name, coordinates, explanation

    except json.JSONDecodeError:
        print("Failed to parse the assistant's reply as valid JSON.")
    except KeyError as e:
        print(f"Missing expected key in JSON response: {e}")



def reasoning_model(text):
    from openai import OpenAI

    with open("/root/UAV-control-using-semantics-and-LLM/config/chatgpt_credentials.yaml", "r") as file:
        data = yaml.safe_load(file)

    client = OpenAI(api_key=data['api_key'])

    prompt = (
            "I am searching for something. "
            "You are a helpful assistant that will parse my description of what i am searching for and give me some help in finding it. "
            "You must always respond with valid JSON containing exactly three keys: "
            "'target object' (the object from COCO dataset I am searching for), 'helper objects' (a list of few objects from MS COCO Dataset),  'explanation' (reasoning why)"
            "Do not include any text outside of the JSON. IMPORTANT - Without Markdown!. "
            "Using the description i gave you, give me a list of objects near which i can find my object. "
            "The objects must only be from a MS COCO dataset and should be written all lower caps. "
            f"Given the following description of what I am searching for: \n{text}\n "
            "in valid JSON, return an answer."
    )

    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
    )

    print(response.choices[0].message.content)
        
if __name__ == '__main__':

    target_class = "Computer mouse"
    #print(define_target_object("I cant find my phone. Usually, I put it to charge when I go to sleep, but I sometimes forget after work."))
    reasoning_model("I cant find my phone. Usually, I put it to charge when I go to sleep, but I sometimes forget after work.")
    
    
