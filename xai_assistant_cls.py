import openai
# from openai.types import *
import base64
# from config import OPENAI_API_KEY
import os
from icecream import ic
import time
import json
from typing import List, Dict
import sys
import numpy as np
from torch import Tensor
from llm_functions import get_shap_values, get_shap_diagram, predict, train_nn_model
import json


class XAIAssistant:
    def __init__(self, assistant_id = None, instruction_additions:dict = None):
        '''
        Initializes the XAIAssistant object. If an assistant_id is provided, the assistant with that id is retrieved. Otherwise, a new assistant is created.
        '''

        self.client = openai.OpenAI()

        self.img_ids = self._create_img_ids()
        instructions = self.prepare_instructions(instruction_additions)
    
        if assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)
            self.update_instructions(instructions)
        else:
            functions_config = self.load_function_config()
            self.assistant = self.client.beta.assistants.create(
            instructions=instructions,
            model="gpt-4o-mini",
            tools=functions_config
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
        base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
        img_paths = [
            os.path.join(base_dir, "images", "image_1.png"),
            os.path.join(base_dir, "images", "image_2.png")
        ]
        img_ids = []
        for img_path in img_paths:
            img = (self.client.files.create(
            file=open(img_path, "rb"),
            purpose="vision"
            ))
            img_ids.append(img.id)
        return img_ids
    def update_instructions(self, instructions):
        self.assistant.instructions = instructions
    def load_function_config(self):
        with open("llm_functions_config.json", "r") as f:
            function_config = json.load(f)
        return function_config
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

        # msgs = self.extract_messages(messages)
        msg = messages.data[0].content[0].text.value
        self.messages.append({"role": "assistant", "content": msg})

        return msg
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


    def tensor_to_list(self, obj):
        if isinstance(obj, Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.tensor_to_list(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self.tensor_to_list(i) for i in obj)
        else:
            return obj
    def serialize_output(self, output):
        # add other cases later
        output = self.tensor_to_list(output)
        output = json.dumps(output)
        return output

    def get_tool_outputs(self, run):
        # Define the list to store tool outputs
        tool_outputs = []

        # Loop through each tool in the required action section
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            args = json.loads(tool.function.arguments)
            
            if tool.function.name == "predict":
                output = predict(
                    age=args.get("age"),
                    sex=args.get("sex"),
                    cp=args.get("cp"),
                    trestbps=args.get("trestbps"),
                    chol=args.get("chol"),
                    fbs=args.get("fbs"),
                    restecg=args.get("restecg"),
                    thalach=args.get("thalach"),
                    exang=args.get("exang"),
                    oldpeak=args.get("oldpeak"),
                    slope=args.get("slope"),
                    ca=args.get("ca"),
                    thal=args.get("thal")
                )
            elif tool.function.name == "get_shap_values":
                output = get_shap_values(
                    age=args.get("age"),
                    sex=args.get("sex"),
                    cp=args.get("cp"),
                    trestbps=args.get("trestbps"),
                    chol=args.get("chol"),
                    fbs=args.get("fbs"),
                    restecg=args.get("restecg"),
                    thalach=args.get("thalach"),
                    exang=args.get("exang"),
                    oldpeak=args.get("oldpeak"),
                    slope=args.get("slope"),
                    ca=args.get("ca"),
                    thal=args.get("thal")
                )
            elif tool.function.name == "get_shap_diagram":
                output = get_shap_diagram(
                    age=args.get("age"),
                    sex=args.get("sex"),
                    cp=args.get("cp"),
                    trestbps=args.get("trestbps"),
                    chol=args.get("chol"),
                    fbs=args.get("fbs"),
                    restecg=args.get("restecg"),
                    thalach=args.get("thalach"),
                    exang=args.get("exang"),
                    oldpeak=args.get("oldpeak"),
                    slope=args.get("slope"),
                    ca=args.get("ca"),
                    thal=args.get("thal"),
                    plot_type=args.get("plot_type", "waterfall")
                )
            elif tool.function.name == "train_nn_model":
                layers_config = args.get("layers_config", [])
                output = train_nn_model(layers_config)
            else:
                output = None

            output = self.serialize_output(output)

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
                        sys.exit(1)
                else:
                    print("No tool outputs to submit.")
            else:
                time.sleep(1)
                print(run.status)


        return self._handle_run_completed(run)
    def get_instructions(self):
        with open("instructions.txt", "r") as f:
            instructions = f.read()
        return instructions

    def prepare_instructions(self, instruction_additions=None):
        instructions = self.get_instructions()
        if instruction_additions:
            for placeholder, instruction_addition in instruction_additions.items():
                position = instructions.find(placeholder)
                instructions = instructions.replace(placeholder, "")
                instructions = instructions[:position] + instruction_addition + instructions[position:]

        return instructions
    
