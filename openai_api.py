import base64
from openai import OpenAI
from config import OPENAI_API_KEY



client = OpenAI(
    api_key=OPENAI_API_KEY
)
# Only needed when file is local. Might be another risk. Lets take web images
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

system_msg = """
You are a Explainer for Explainable AI. 
The user will ask questions about the provided image, which is a explainable AI Visualizations like SHAP or LIME. 
You only answer questions to this topic. Otherwise you say 'This is not my task'
"""
completion = client.chat.completions.create(
  model="gpt-4-turbo-2024-04-09",
  # model = "gpt-3.5-turbo-0125",
  # stream=True,
  messages=[
    {"role": "system", "content": system_msg},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://images.datacamp.com/image/upload/v1688055328/image_c957bbcce7.png",
          },
        },
      ],
    }
  ]
)

print(completion.choices[0].message)

"""
Result:
ChatCompletionMessage(content='This image is a bar chart representing SHAP (SHapley Additive exPlanations) values, which help in explaining the output of a model. The chart displays different variables or features on the y-axis (from "Tariff Plan" to "Status") and their respective mean SHAP values on the x-axis, indicating the average impact of these features on the prediction 
of two classes (Class 0 and Class 1).\n\nIn this chart:\n- Blue bars represent the impact of each feature on Class 0.\n- Red bars represent the impact of each feature on Class 1.\n\nThe length of each bar indicates the magnitude of the feature\'s impact on model outcomes. Positive values suggest a higher likelihood or boost towards the respective class, whereas negative values 
suggest a decrease. For example, a longer blue bar for "Status" suggests a significant positive impact on predicting Class 0 when the status feature has specific values. In contrast, the respective shorter red bar suggests a lesser positive impact on Class 1 for the same feature\'s value.\n\nThis kind of visualization is very useful to understand which features are most influential in the predictions of a model and how different feature values impact the prediction of different classes.', role='assistant', function_call=None, tool_calls=None)
"""


# messages=[
#     {
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "What’s in this image?"},
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#           },
#         },
#       ],
#     }
#   ],