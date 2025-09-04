from ingest_data import fetch_data
from segmentation import segment_text_batch
from sample import create_sample_for_annotation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import spacy
import csv

DATA_FOLDER = "data"
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, "processed")
DATASET_NAME = "louisbrulenaudet/clinical-trials"
EDA_PLOT_FILE = os.path.join(DATA_FOLDER, "criteria_length_distribution.png")

COLUMNS_TO_KEEP = [
    'nct_id', 'eligibility_criteria', 'overall_status', 'phases', 'study_type',
    'minimum_age', 'maximum_age', 'sex', 'healthy_volunteers', 'conditions',
    'keywords', 'interventions', 'mesh_terms', 'locations', 'brief_title',
    'official_title', 'brief_summary'
]

def main():
    print("--- Phase I: Data Preprocessing ---")
    try:
        nlp = spacy.load("en_core_sci_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        nlp.add_pipe("sentencizer")
        print("SciSpacy model loaded successfully.")
    except OSError:
        print("SciSpacy model not found. Please run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz")
        return


    ds = fetch_data(DATASET_NAME)
    split = ds['train'].train_test_split(test_size=0.1, seed=42)
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    for split_name in split.keys():
        print(f"\n--- Processing '{split_name}' data ---")
        df = split[split_name].to_pandas()
        df = df[COLUMNS_TO_KEEP]
        df.dropna(subset=['eligibility_criteria'], inplace=True)
        print("   - Columns filtered and invalid rows dropped.")

        if split_name == "train":
            df['criteria_length'] = df['eligibility_criteria'].str.len()
            plt.figure(figsize=(12, 6))
            sns.histplot(df['criteria_length'].dropna(), bins=50, kde=True)
            plt.title('Distribution of Eligibility Criteria Text Length (Train Set)')
            plt.savefig(EDA_PLOT_FILE)
            plt.close()
            print(f"   - EDA plot saved to '{EDA_PLOT_FILE}'.")

        print(f"   - Starting segmentation for {len(df)} documents...")
        
        criteria_list = df['eligibility_criteria'].tolist()
        with tqdm(total=1, desc=f"Segmenting {split_name} data") as pbar:
            segmented_results = segment_text_batch(criteria_list, nlp)
            pbar.update(1)
            
        df['segmented_criteria'] = segmented_results
        print("   - Segmentation complete.....")
        print("   - Final Transformation.....")
        df_exploded = df.explode('segmented_criteria').rename(columns={'segmented_criteria': 'criterion_text'})
        df_exploded.dropna(subset=['criterion_text'], inplace=True)
        final_columns = [col for col in df.columns if col not in ['eligibility_criteria', 'criteria_length', 'segmented_criteria']]
        final_columns.insert(1, 'criterion_text')
        df_final = df_exploded[final_columns].copy()
        output_path = os.path.join(PROCESSED_DATA_FOLDER, f"{split_name}.csv")
        df_final.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"   - {split} saved to'{output_path}'.")
    create_sample_for_annotation()


if __name__ == "__main__":
    main()