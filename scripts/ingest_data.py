import os
from datasets import load_dataset, load_from_disk, get_dataset_split_names

def fetch_data(dataset_name: str, output_folder: str = "../data"):
    dataset_path = os.path.join(output_folder, dataset_name.replace("/", "_"))
    if not os.path.exists(dataset_path):
        os.makedirs(output_folder, exist_ok=True)
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(dataset_path)
    else:
        dataset = load_from_disk(dataset_path)

    return dataset
