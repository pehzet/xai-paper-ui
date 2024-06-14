responses = {'q8': 'Disagree strongly', 'I see myself as disorganized, careless.': 'Neither agree nor disagree', 'age': 18, 'FormSubmitter:my_form-Submit': True, 
'I see myself as reserved, quiet.': 'Agree moderately', 'q9': 'Disagree strongly', 'I see myself as open to new experiences, complex.': 'Agree a little', 'q4': 'Disagree strongly', 'q1': 'Disagree strongly', 'q2': 'Disagree strongly', 'name': '', 'q3': 'Disagree strongly', 'q5': 'Disagree strongly', 'I see myself as sympathetic, warm.': 'Disagree a little', 'I see myself as dependable, self-disciplined.': 'Disagree a little', 'survey_completed': 
True, 'I see myself as critical, quarrelsome.': 'Neither agree nor disagree', 'I see myself as conventional, uncreative.': 'Neither agree nor disagree', 'I see myself as calm, emotionally stable.': 'Neither agree nor disagree', 'q6': 'Disagree strongly', 'I see myself as anxious, easily upset.': 'Neither agree nor disagree', 'I see myself as extraverted, enthusiastic.': 'Disagree moderately', 'q10': 'Disagree strongly', 'q7': 'Disagree strongly'}
responses2 = {'q8': 'Neither agree nor disagree', 'I see myself as disorganized, careless.': 'Disagree moderately', 'age': 18, 'FormSubmitter:my_form-Submit': True, 'I see myself as reserved, quiet.': 'Disagree moderately', 'q9': 'Neither agree nor disagree', 'I see myself as open to new experiences, complex.': 'Agree moderately', 'q4': 'Neither agree nor disagree', 'q1': 'Disagree moderately', 'q2': 'Neither agree nor disagree', 'name': 'MoinMeister', 
'q3': 'Disagree a little', 'q5': 'Agree a little', 'I see myself as sympathetic, warm.': 'Agree moderately', 'I see myself as dependable, self-disciplined.': 'Disagree a little', 'survey_completed': True, 'I see myself as critical, quarrelsome.': 'Neither agree nor disagree', 'I see myself as conventional, uncreative.': 'Agree moderately', 'I see myself as calm, emotionally stable.': 'Agree a little', 'q6': 'Agree moderately', 'I see myself as anxious, easily upset.': 'Disagree moderately', 'I see myself as extraverted, enthusiastic.': 'Agree a little', 'q10': 'Neither agree nor disagree', 'q7': 'Disagree a little'}
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
    average_scores = {trait: sum(values) / len(values) for trait, values in scores.items()}

    return average_scores

# calculate_scores(responses2)