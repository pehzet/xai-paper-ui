import pandas as pd
import os
# Lade den Datensatz
file_name = 'Crop_recommendation.csv'
file_folder = 'data'
file_path =  os.path.join(os.path.dirname(__file__), file_folder, file_name)
data = pd.read_csv(file_path)

# Beschreibung des Datensatzes
def describe_dataset(df):
    description = {}
    description['Number of Entries'] = len(df)
    description['Columns'] = df.columns.tolist()
    description['Label Column'] = 'label' if 'label' in df.columns else None
    description['Number of Labels'] = df['label'].nunique() if 'label' in df.columns else None
    description['Label Distribution'] = df['label'].value_counts().to_dict() if 'label' in df.columns else None
    description['Numeric Columns'] = df.select_dtypes(include=['number']).columns.tolist()
    description['Ranges of Numeric Columns'] = {
        col: (df[col].min(), df[col].max()) for col in df.select_dtypes(include=['number']).columns
    }
    description['Missing Values'] = df.isnull().sum().to_dict()
    
    return description

# Beschreibung ausgeben
dataset_description = describe_dataset(data)
with open("dataset_description.txt", "w") as f:
    for key, value in dataset_description.items():
        f.write(f"{key}: {value}\n")
# Ausgabe
for key, value in dataset_description.items():
    print(f"{key}: {value}")
