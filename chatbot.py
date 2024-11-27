from openai import OpenAI
import config
import json
import os
import base64
import sys
from icecream import ic
from prediction_model.model_interface import predict
from prediction_model.shap_interface import predict_shap_values, generate_shap_diagram
class XAIChatbot:
    def __init__(self):
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.MODEL
        self.messages = []
        self.messages.append(self.create_instruction_message())
        self.messages.append(self.create_img_message())
        self.function_config = self.load_function_config()
        self._tool_call_id_with_image = None
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def load_function_config(self):
        with open("function_config.json", "r") as f:
            function_config = json.load(f)
        return function_config
    def create_instruction_message(self):
        with open("instructions.txt", "r", encoding="utf-8") as f:
            message = f.read()
        return {
            "role": "system",
            "content": message
        }
    def get_messages(self):
        return self.messages
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
            if tool_call.function.name == "predict":
                output = predict(tool_call.function.arguments)
            elif tool_call.function.name == "generate_shap_diagram":
                output = generate_shap_diagram(tool_call.function.arguments)
                output = output.get("shap_diagram")
                self._tool_call_id_with_image = tool_call.id
            # elif tool_call.function.name == "predict_shap_values":
            #     output = predict_shap_values(tool_call.function.arguments)
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

        return msgs
    def get_completion(self):
        print("Creating completion")
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.function_config
        )
    def add_tool_call_prior_response_to_messages(self, response):
        tool_call_array = []
        for tool_call in response.tool_calls:
            tool_call_array.append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        obj = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_call_array
            }
        self.messages.append(obj)
    def chat(self,msg):
        message = self._create_message("user", msg)
        self.messages.append(message)
        completion = self.get_completion()
        response = completion.choices[0].message
        ic(response)
        image = None
        if response.tool_calls:
            self.add_tool_call_prior_response_to_messages(response)
            tool_outputs = self.handle_tool_calls(response.tool_calls)
            tool_msgs = self.create_tool_messages(tool_outputs)
            # self.messages.append(response)
            for tool_msg in tool_msgs:
                
                self.messages.append(tool_msg)
                if tool_msg["tool_call_id"] == self._tool_call_id_with_image:
                    image = tool_msg["content"]
                    self._tool_call_id_with_image = None

            # ic(self.messages[2].content[0])
            completion = self.get_completion()
            ic(completion)
            response = completion.choices[0].message

       
        response_oai = self._create_message("assistant", response.content)
        self.messages.append(response_oai)
        # ic(self.messages)
        return response.content, image
