import openai
# from openai.types import *
import base64
from config import OPENAI_API_KEY
from icecream import ic
import time
import json
from typing import List, Dict
import sys
from model_interface import predict, explain

class XAIAssistant:
    def __init__(self, assistant_id = None):
        '''
        Initializes the XAIAssistant object. If an assistant_id is provided, the assistant with that id is retrieved. Otherwise, a new assistant is created.
        '''
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.img_ids = self._create_img_ids()

        if assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)
        else:
            with open("instructions.txt", "r", encoding="utf-8") as f:
                instructions = f.read()
            self.assistant = self.client.beta.assistants.create(
            instructions=instructions,
            model="gpt-4o",
            tools=[
                {
                "type": "function",
                "function": {
                    "name": "predict",
                    "description": "Get the predicted value for a specific input. If the user only provides one of the three inputs, the other two should be set to -99.",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {
                        "type": "integer",
                        "description": "Value of the x1 input. Can be between 0 and 100."
                        },
                        "x2": {
                        "type": "integer",
                        "description": "Value of the x2 input. Can be between 0 and 100."
                        },
                        "x3": {
                        "type": "integer",
                        "description": "Value of the x1 input. Can be between 0 and 100."
                        },
                    },
                    "required": ["x1", "x2", "x3"]
                    }
                }
                },
                {
                "type": "function",
                "function": {
                    "name": "explain",
                    "description": "Get the explaination for a specific input. If the user only provides one of the three inputs, the other two should be set to None. If they are None the previous values will be used",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {
                        "type": "integer",
                        "description": "Value of the x1 input. Can be between 0 and 100."
                        },
                        "x2": {
                        "type": "integer",
                        "description": "Value of the x2 input. Can be between 0 and 100."
                        },
                        "x2": {
                        "type": "integer",
                        "description": "Value of the x1 input. Can be between 0 and 100."
                        },
                    },
                    "required": ["x1", "x2", "x3"]
                    }
                }
                }
            ]
    
            )
        msg_objects = [
                {
                    "role": "user",
                    "content": [

                     {"type": "text", "text": "Hier sind die beiden Bilder mit SHAP Werten. Gehe vorerst nicht drauf ein. Begrüße mich einfach mit Hallo."},
                    {
                    "type": "image_file",
                    "image_file": {"file_id": self.img_ids[0]}
                    },
                    {
                    "type": "image_file",
                    "image_file": {"file_id": self.img_ids[1]}
                    },

                    ]
        
                }
            ]

        self.thread = self.client.beta.threads.create(
     
            messages=msg_objects
                
            
        )
        self.messages = []
    def _create_img_ids(self):
        img_paths = [
                r"images\image_1.png",
                r"images\image_2.png"
            ]
        img_ids = []
        for img_path in img_paths:
            img = (self.client.files.create(
            file=open(img_path, "rb"),
            purpose="vision"
            ))
            img_ids.append(img.id)
        return img_ids

    def _create_message(self,role,content):
        return self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role=role,
            content=content
        )
    def _create_run(self):
        #runs last 10 minutes
        return self.client.beta.threads.runs.create_and_poll(
        thread_id=self.thread.id,
        assistant_id=self.assistant.id,
        )
    def _handle_run_completed(self, run):
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )
        
        
        msgs = self.extract_messages(messages)

        self.messages.append({"role": "assistant", "content": messages.data[0].content[0].text.value})


        return messages.data[0].content[0].text.value
    def extract_messages(self, messages: List) -> List[Dict[str, str]]:
        result = []
  
        for msg in messages:

            content_text = " ".join(block.text.value for block in msg.content if block.type == "text")
            cleaned_content = self.clean_latex_formatting(content_text)
            result.append({
                "role": msg.role,
                "content": cleaned_content
                })
        return result

    def clean_latex_formatting(self,text: str) -> str:
        # Entfernt alle LaTeX-Mathematik-Umgebungen
        cleaned_text = text.replace("\\(", "").replace("\\)", "")
        return cleaned_text

    def _predict(self, x1, x2, x3):
        if x1 is None or x1 == -99:
            x1 = 1
        if x2 is None or x2 == -99:
            x2 = 2
        if x3 is None or x3 == -99:
            x3 = 3
        return predict(x1, x2, x3)

    def get_tool_outputs(self, run):
        # Define the list to store tool outputs
        tool_outputs = []
        
        # Loop through each tool in the required action section
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "predict":
           
                args = json.loads(tool.function.arguments)
 
                output = self._predict(args["x1"],args["x2"],args["x3"])
                tool_outputs.append({
                "tool_call_id": tool.id,
                "output": output
                })
            elif tool.function.name == "explain":
                args = json.loads(tool.function.arguments)
       
                output = explain(args["x1"],args["x2"],args["x3"])
                tool_outputs.append({
                "tool_call_id": tool.id,
                "output": output
                })
        return tool_outputs

    def _get_encoded_image(self, image_idx):
        img_paths = [
            r"images\image_1.png",
            r"images\image_2.png"
        ]
        try:
            img_path = img_paths[image_idx]
        except IndexError:
            return None
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
        

    def chat(self,msg):
        message = self._create_message("user", msg)
        run = self._create_run()
        while not run.status == 'completed':

            if run.status == 'requires_action':
                tool_outputs = self.get_tool_outputs(run)
                if tool_outputs:
                    try:
                        run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=self.thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                        )
                        print("Tool outputs submitted successfully.")
                    except Exception as e:
                        print("Failed to submit tool outputs:", e)
                else:
                    print("No tool outputs to submit.")
            else:
                time.sleep(1)
                print(run.status)


        return self._handle_run_completed(run)