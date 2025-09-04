import pandas as pd
import os

PROCESSED_DATA_FOLDER = os.path.join("data", "processed")
PROCESSED_TRAIN_FILE = os.path.join(PROCESSED_DATA_FOLDER, "train.csv")
ANNOTATION_SAMPLE_FILE = os.path.join(PROCESSED_DATA_FOLDER, "sample.txt")
SAMPLE_SIZE = 1000

def create_sample_for_annotation():
    if not os.path.exists(PROCESSED_TRAIN_FILE):
        return

    try:
        df = pd.read_csv(PROCESSED_TRAIN_FILE)
    except Exception as e:
        print(f"Failed to load the CSV file. Reason: {e}")
        return

    if 'criterion_text' not in df.columns:
        print("'criterion_text' column not found in the CSV.")
        return

    unique_criteria = df['criterion_text'].dropna().unique()
    if len(unique_criteria) < SAMPLE_SIZE:
        sample_size_to_take = len(unique_criteria)
    else:
        sample_size_to_take = SAMPLE_SIZE

    sample_criteria = unique_criteria[:sample_size_to_take]
    with open(ANNOTATION_SAMPLE_FILE, 'w', encoding='utf-8') as f:
        for text in sample_criteria:
            f.write(f"{text.strip()}\n")
        
        print(f"\nAnnotation sample file created successfully!")

if __name__ == "__main__":
    create_sample_for_annotation()