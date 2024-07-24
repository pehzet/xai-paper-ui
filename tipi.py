# responses = {'q8': 'Disagree strongly', 'I see myself as disorganized, careless.': 'Neither agree nor disagree', 'age': 18, 'FormSubmitter:my_form-Submit': True, 
# 'I see myself as reserved, quiet.': 'Agree moderately', 'q9': 'Disagree strongly', 'I see myself as open to new experiences, complex.': 'Agree a little', 'q4': 'Disagree strongly', 'q1': 'Disagree strongly', 'q2': 'Disagree strongly', 'name': '', 'q3': 'Disagree strongly', 'q5': 'Disagree strongly', 'I see myself as sympathetic, warm.': 'Disagree a little', 'I see myself as dependable, self-disciplined.': 'Disagree a little', 'survey_completed': 
# True, 'I see myself as critical, quarrelsome.': 'Neither agree nor disagree', 'I see myself as conventional, uncreative.': 'Neither agree nor disagree', 'I see myself as calm, emotionally stable.': 'Neither agree nor disagree', 'q6': 'Disagree strongly', 'I see myself as anxious, easily upset.': 'Neither agree nor disagree', 'I see myself as extraverted, enthusiastic.': 'Disagree moderately', 'q10': 'Disagree strongly', 'q7': 'Disagree strongly'}
# responses2 = {'q8': 'Neither agree nor disagree', 'I see myself as disorganized, careless.': 'Disagree moderately', 'age': 18, 'FormSubmitter:my_form-Submit': True, 'I see myself as reserved, quiet.': 'Disagree moderately', 'q9': 'Neither agree nor disagree', 'I see myself as open to new experiences, complex.': 'Agree moderately', 'q4': 'Neither agree nor disagree', 'q1': 'Disagree moderately', 'q2': 'Neither agree nor disagree', 'name': 'MoinMeister', 
# 'q3': 'Disagree a little', 'q5': 'Agree a little', 'I see myself as sympathetic, warm.': 'Agree moderately', 'I see myself as dependable, self-disciplined.': 'Disagree a little', 'survey_completed': True, 'I see myself as critical, quarrelsome.': 'Neither agree nor disagree', 'I see myself as conventional, uncreative.': 'Agree moderately', 'I see myself as calm, emotionally stable.': 'Agree a little', 'q6': 'Agree moderately', 'I see myself as anxious, easily upset.': 'Disagree moderately', 'I see myself as extraverted, enthusiastic.': 'Agree a little', 'q10': 'Neither agree nor disagree', 'q7': 'Disagree a little'}

from openai import OpenAI
from icecream import ic
def reverse_score(score):
    # This function reverses the Likert scale score for negatively worded questions
    return 8 - score

def calculate_scores(responses):
    # Scores dictionary
    scores = {
        "Extraversion": [],
        "Agreeableness": [],
        "Conscientiousness": [],
        "Neuroticism": [],
        "Openness": []
    }

    # Mapping questions to their respective traits
    # question_traits = {
    #     "I see myself as extraverted, enthusiastic.": "Extraversion",
    #     "I see myself as critical, quarrelsome.": "Agreeableness",
    #     "I see myself as dependable, self-disciplined.": "Conscientiousness",
    #     "I see myself as anxious, easily upset.": "Neuroticism",
    #     "I see myself as open to new experiences, complex.": "Openness",
    #     "I see myself as reserved, quiet.": "Extraversion",
    #     "I see myself as sympathetic, warm.": "Agreeableness",
    #     "I see myself as disorganized, careless.": "Conscientiousness",
    #     "I see myself as calm, emotionally stable.": "Neuroticism",
    #     "I see myself as conventional, uncreative.": "Openness"
    # }
    question_traits = {
        "q1": "Extraversion",
        "q2": "Agreeableness",
        "q3": "Conscientiousness",
        "q4": "Neuroticism",
        "q5": "Openness",
        "q6": "Extraversion",
        "q7": "Agreeableness",
        "q8": "Conscientiousness",
        "q9": "Neuroticism",
        "q10": "Openness"
    }
    options = [
        "Disagree strongly",
        "Disagree moderately",
        "Disagree a little",
        "Neither agree nor disagree",
        "Agree a little",
        "Agree moderately",
        "Agree strongly"
    ]

    # Calculate scores for each trait based on responses
    for question, trait in question_traits.items():
        # response = st.session_state[question]
        response = responses[question]

        score = options.index(response) + 1

        # if question in ["I see myself as critical, quarrelsome.",
        #                 "I see myself as reserved, quiet.",
        #                 "I see myself as disorganized, careless.",
        #                 "I see myself as anxious, easily upset.",
        #                 "I see myself as conventional, uncreative."]:
        if question in ["q2", "q4", "q6", "q8", "q10"]:
            score = reverse_score(score)
        scores[trait].append(score)

    # Average scores for each trait
    average_scores = {trait: round(sum(values) / len(values)) for trait, values in scores.items()}

    return average_scores


def get_description_from_scores(scores):
    from config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I will give you tipi scores and i want you to write a description of the person in the third person ('The Person is ...'). Tipi is a personality test that measures the Big Five personality traits. The Big Five personality traits are Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness. The scores range from 1 to 7. The higher the score, the more you exhibit that trait. For example, if you have a high score in Extraversion, you are outgoing and social. If you have a low score in Neuroticism, you are calm and emotionally stable. The scores are calculated based on the responses of previous survey questions. Here are the scores: " + str(scores)},

    ]
    )

    return response.choices[0].message.content
