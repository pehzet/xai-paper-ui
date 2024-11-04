from openai import OpenAI
import config
import json
import os
import base64
import sys
from icecream import ic
class XAIChatbot:
    def __init__(self):
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.MODEL
        self.messages = []
        self.messages.append(self.create_instruction_message())
        self.messages.append(self.create_img_message())
        self.function_config = self.load_function_config()
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def load_function_config(self):
        with open("function_config.json", "r") as f:
            function_config = json.load(f)
        return function_config
    def create_instruction_message(self):
        with open("instructions.txt", "r") as f:
            message = f.read()
        return {
            "role": "system",
            "content": message
        }
    def create_img_message(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
        image1_path = os.path.join(base_dir, "images", "global_explanation.png")
        image2_path = os.path.join(base_dir, "images", "local_explanation.png")
        img1 = self.encode_image(image1_path)
        img2 = self.encode_image(image2_path)
        message = {
            "role": "user",
            "content": [
        {
          "type": "text",
          "text": "Hier sind die beiden Bilder mit SHAP Werten. Das erste ist eine globale Erklärung und das zweite eine lokale. Gehe vorerst nicht drauf ein. Erst wenn ich explizit danach frage.",
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/png;base64,{img1}"
          },
        },
               {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/png;base64,{img2}"
          },
        },
      ],
        }
        return message
    def _create_message(self, role, content):
        return {
            "role": role,
            "content": content
        }
    def handle_tool_calls(self, tool_calls):
        tool_outputs = []
        for tool_call in tool_calls:
            if tool_call.function.name == "get_shap_diagram":
                output = self.dummy_function_image()
            elif tool_call.function.name == "get_shap_values":
                output = self.dummy_function_values()
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "output": output
            })
        return tool_outputs
    def create_tool_messages(self, tool_outputs):
        msgs = []
        for output in tool_outputs:
            msgs.append({
                "role": "tool",
                "content": output["output"],
                "tool_call_id": output["tool_call_id"]

            })
        ic(msgs)
        return msgs
    def get_completion(self):
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.function_config
        )
    
    def chat(self,msg):
        ic(msg)
        message = self._create_message("user", msg)
        self.messages.append(message)
        completion = self.get_completion()
        ic(completion)
        response = completion.choices[0].message
        
        if response.tool_calls:
            tool_outputs = self.handle_tool_calls(response.tool_calls)
            ic(tool_outputs)
            tool_msgs = self.create_tool_messages(tool_outputs)
            ic(response)
            self.messages.append(response)
            for tool_msg in tool_msgs:
                self.messages.append(tool_msg)
            completion = self.get_completion()
            response = completion.choices[0].message


        response_oai = self._create_message("assistant", response.content)
        self.messages.append(response_oai)
        return response.content

    def dummy_function_image(self):
        pth = os.path.join("images", "global_explanation.png")
        img = self.encode_image(pth)
        return img
    def dummy_function_values(self):
        features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        values = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
        return dict(zip(features, values))