import pandas as pd

# Daten laden
data = pd.read_csv('Crop_recommendation.csv')

# # Anzahl der Elemente in Gruppen
# group_counts = data['label'].value_counts()
# print(group_counts)
'''
label
rice          139
Soyabeans     130
banana        130
beans         125
cowpeas       122
orange        122
maize         119
coffee        110
peas          100
groundnuts    100
mango         100
watermelon    100
grapes        100
apple         100
cotton        100
'''
# Mittelwerte für jede Gruppe berechnen
label_means = data.groupby('label').mean(numeric_only=True)

# Funktion, um den nächsten Eintrag zum Mittelwert zu finden
def find_closest_to_mean(group):
    mean_values = label_means.loc[group.name]
    distances = ((group.iloc[:, :-1] - mean_values) ** 2).sum(axis=1)
    closest_index = distances.idxmin()
    return group.loc[closest_index]

# Auf Gruppen anwenden, um repräsentative Einträge zu finden
representative_closest_entries = data.groupby('label').apply(find_closest_to_mean).reset_index(drop=True)

# Repräsentative Einträge anzeigen
print(representative_closest_entries)
representative_closest_entries.to_csv('test_cases.csv', index=False)
