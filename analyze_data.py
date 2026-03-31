import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Crop_recommendation.csv')

print("Dataset Dimensions:", df.shape)
print("\nChecking for Missing Data:\n", df.isnull().sum())

numeric_df = df.drop('label', axis=1)

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap of Soil Features")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='label', y='rainfall', data=df)
plt.xticks(rotation=90)
plt.title("Average Rainfall Required per Crop")
plt.show()