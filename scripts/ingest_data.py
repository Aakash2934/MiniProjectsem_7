import os
from datasets import load_dataset, load_from_disk, get_dataset_split_names
from tqdm.auto import tqdm

dataset_name = "louisbrulenaudet/clinical-trials"
output_folder = "../data"
dataset_path = os.path.join(output_folder, "clinical_trials_dataset")
if not os.path.exists(dataset_path):
    os.makedirs(output_folder, exist_ok=True)
    splits = get_dataset_split_names(dataset_name)
    print("Available splits:", splits)
    clinical_trials_dataset = load_dataset(dataset_name)
    clinical_trials_dataset.save_to_disk(dataset_path)
else:
    clinical_trials_dataset = load_from_disk(dataset_path)
