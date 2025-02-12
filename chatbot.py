from openai import OpenAI
import toml
import json
import os
import base64
import sys
from prediction_model.model_interface import predict, sum_feature, mean_feature, quantile_feature, variance_feature, std_feature, min_feature, max_feature, correlation, class_distribution, feature_values_for_class, feature_distribution, run_simulation
from prediction_model.shap_interface import predict_shap_values, generate_shap_diagram
import pandas as pd
import streamlit as st
class XAIChatbot:
    def __init__(self, decision_no=None):
        # config_path = os.path.join(".streamlit", "config.toml")
        config = st.secrets #toml.load(config_path)
        config = config.get("llm")
        self.client = OpenAI(api_key=config.get("OPENAI_API_KEY"))
        self.decision_no = decision_no
        self.model = config.get("MODEL")
        self.init_messages()
        self.function_config = self.load_function_config()
        self._tool_call_id_with_image = None
    def _get_decision_case(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        decisions_file = os.path.join(current_dir, "prediction_model","data", "test_cases.csv")
        decisions = pd.read_csv(decisions_file)
        decisions = decisions.drop("label", axis=1)
        return json.dumps(decisions.iloc[self.decision_no-1].to_dict())

    def init_messages(self):
        self.messages = []
        decision_values = self._get_decision_case()
        instruction_message = self.create_instruction_message(placeholder="{{ decision_values }}", placeholder_value=decision_values)
        self.messages.append(instruction_message)
        self.messages.append(self.create_img_message())

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def load_function_config(self):
        with open("function_config.json", "r") as f:
            function_config = json.load(f)
        return function_config
    def create_instruction_message(self, placeholder:str=None, placeholder_value:str=None):
        with open("instructions.txt", "r", encoding="utf-8") as f:
            message = f.read()
        if placeholder:
            message = message.replace(placeholder, placeholder_value)
        return {
            "role": "system",
            "content": message
        }
    def get_messages(self):
        return self.messages
    def create_img_message(self, decision_no=None):
        if decision_no is None:
            file_name_1 = "global_explanation.png"
            file_name_2 = "local_explanation.png"
        else:
            file_name_1 = f"case{decision_no}_global.png"
            file_name_2 = f"case{decision_no}_local.png"
        base_dir = os.path.dirname(os.path.abspath(__file__))  # aktuelles Verzeichnis
        image1_path = os.path.join(base_dir, "images", file_name_1)
        image2_path = os.path.join(base_dir, "images", file_name_2)
        img1 = self.encode_image(image1_path)
        img2 = self.encode_image(image2_path)
        message = {
            "role": "user",
            "content": [
        {
          "type": "text",
          "text": "Here are the two images with SHAP values. The first is a global explanation and the second is a local one. Don't go into it for now. Only when I explicitly ask for it.",
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
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments
            if fn_name == "predict":
                output = predict(fn_args)

            elif fn_name == "generate_shap_diagram":
                # Keep the SHAP diagram logic example, ignoring actual implementation details
                shap_result = generate_shap_diagram(fn_args)
                # If the function returns a dict with a key "shap_diagram", use that
                if isinstance(shap_result, dict):
                    output = shap_result.get("shap_diagram")
                else:
                    # Otherwise just return the whole result
                    output = shap_result
                # Remember the ID in case you need to reference it for images
                self._tool_call_id_with_image = tool_call.id
            elif fn_name == "sum_feature":
                output = sum_feature(fn_args)

            elif fn_name == "mean_feature":
                output = mean_feature(fn_args)

            elif fn_name == "quantile_feature":
                output = quantile_feature(fn_args)

            elif fn_name == "variance_feature":
                output = variance_feature(fn_args)

            elif fn_name == "std_feature":
                output = std_feature(fn_args)

            elif fn_name == "min_feature":
                output = min_feature(fn_args)

            elif fn_name == "max_feature":
                output = max_feature(fn_args)

            elif fn_name == "correlation":
                output = correlation(fn_args)

            elif fn_name == "class_distribution":
                output = class_distribution(fn_args)

            elif fn_name == "feature_values_for_class":
                output = feature_values_for_class(fn_args)

            elif fn_name == "feature_distribution":
                output = feature_distribution(fn_args)

            elif fn_name == "run_simulation":
                output = run_simulation(fn_args)
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
                "content": str(output["output"]),
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


            completion = self.get_completion()

            response = completion.choices[0].message

       
        response_oai = self._create_message("assistant", response.content)
        self.messages.append(response_oai)

        return response.content, image
