import pandas as pd

# Load the dataset
df = pd.read_csv("../data/fake_and_real_news.csv")

# Display the first 5 rows
print(df.head())

# Show columns and dataset size
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))

# Distribution of labels
print(df['label'].value_counts())
