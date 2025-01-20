import os
import pandas as pd
from shap_utils import SHAPUtility
from neuronal_network import CropPredictor

def create_shap_diagrams(test_cases_file='data/test_cases.csv', output_dir='images'):
    """
    Generate SHAP diagrams for test cases and save them as images.

    Parameters:
    - test_cases_file (str): Path to the test cases file (CSV).
    - output_dir (str): Directory to save the output images.
    """
    # Load test cases
    test_cases = pd.read_csv(test_cases_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the predictor and SHAP explainer
    predictor = CropPredictor()
    explainer = SHAPUtility.initialize_shap_explainer(predictor)

    # Iterate over test cases
    for idx, case in test_cases.iterrows():
        # Extract case ID and features
        test_case_id = idx + 1  # Assuming 1-based indexing for IDs
        if test_case_id != 13:
            continue
        label = case['label']
        features = case.drop('label').to_dict()

        # Global SHAP explanation for the label
        global_file = os.path.join(output_dir, f"case{test_case_id}_global.png")
        SHAPUtility.get_shap_global_explanation(predictor, explainer, label)
        SHAPUtility._save_plot(global_file)

        # Local SHAP explanation for the test case
        local_file = os.path.join(output_dir, f"case{test_case_id}_local.png")
        input_data = {**features, "plot_type": "waterfall"}
        SHAPUtility.generate_shap_diagram(input_data)
        SHAPUtility._save_plot(local_file)

        print(f"Generated SHAP diagrams for test case {test_case_id}:")
        print(f" - Global: {global_file}")
        print(f" - Local: {local_file}")

if __name__ == "__main__":
    # Run the function
    create_shap_diagrams()
