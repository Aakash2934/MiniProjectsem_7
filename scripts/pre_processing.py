from ingest_data import fetch_data
from segmentation import segmentation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os



def main():
    print("Fetching Data\n")
    dataset_name = "louisbrulenaudet/clinical-trials"
    try:
        ds = fetch_data(dataset_name)
        print("Data Downloded and Saved Sucessfully\n")
    except Exception as e:
        print(f"An error occurred while fetching the dataset: {e}\n")
        return

    tqdm.pandas(desc="Segmenting Criteria")
    split = ds['train'].train_test_split(test_size=0.1, seed=42)
    os.makedirs("../data/pro", exist_ok=True)
    DATA_FOLDER = "data/pro"

    for i in split.keys():
        print(f"=================Performing pre-processing task on {i} data=======================\n")
        df=split[i].to_pandas()
        df=df[1:3]
        COLUMNS_TO_KEEP = [
                            'nct_id', 'eligibility_criteria', 'overall_status', 'phases', 'study_type',
                            'minimum_age', 'maximum_age', 'sex', 'healthy_volunteers', 'conditions',
                            'keywords', 'interventions', 'mesh_terms', 'locations', 'brief_title',
                            'official_title', 'brief_summary'
                        ]
        df=df[COLUMNS_TO_KEEP]
        print(f"Removed unwanted columns from {i} data\n")
        df.dropna(subset=['eligibility_criteria'], inplace=True)
        print(f"Removed missing value row with respect to eligibility_criteria from {i} data\n")

        if i == "train":
            df['criteria_length'] = df['eligibility_criteria'].str.len()
            plt.figure(figsize=(12, 6))
            sns.histplot(df['criteria_length'].dropna(), bins=50, kde=True)
            plt.title('Distribution of Eligibility Criteria Text Length)')
            plt.xlabel('No of Char')
            plt.ylabel('No of Trials')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

        print(f"Starting Criterion Segmentation of {i} data\n")
        try:
            df=segmentation(df)
            print(f"Criterion Segmentation of {i} data is sucessfull\n")
        except Exception as e:
            print(f"An error occurred while segmentation the dataset: {e}\n")
            return
        
        print(f"Performing final transform and Save {i} data\n")
        df_exploded = df.explode('segmented_criteria').rename(columns={'segmented_criteria': 'criterion_text'})
        df_exploded.dropna(subset=['criterion_text'], inplace=True)
        final_columns = [col for col in df.columns if col not in ['eligibility_criteria', 'criteria_length', 'segmented_criteria']]
        final_columns.insert(1, 'criterion_text')
        df_final = df_exploded[final_columns].copy()
        output_path = os.path.join(DATA_FOLDER, f"{i}.csv")
        df_final.to_csv(output_path, index=False)
        print(f"{i} data has been pre-processed")


if __name__ == "__main__":
    main()
