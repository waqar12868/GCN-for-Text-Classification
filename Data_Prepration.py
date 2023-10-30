import re
import pandas as pd
# Read the CSV file into a DataFrame
df = pd.read_csv("file_name.csv")
# Show the first few rows of the DataFrame
print(df.head())

def clean_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower().split()  # Convert to lowercase and tokenize
    return text
# Apply text preprocessing
df['Message_clean'] = df['Message'].apply(clean_and_tokenize)
# Show the updated DataFrame
print(df.head())
