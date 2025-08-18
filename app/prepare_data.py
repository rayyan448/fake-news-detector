import pandas as pd

df = pd.read_csv("../data/fake_and_real_news.csv")

# Show how many missing labels
print("Missing labels:", df['label'].isnull().sum())
print("Total rows:", len(df))

# Drop rows where label is missing
df = df.dropna(subset=['label'])

# Show value counts
print(df['label'].value_counts())

# Optional: Clean missing text headlines if needed
df = df.dropna(subset=['Text'])

# Sample 5 rows
print(df.sample(5))
