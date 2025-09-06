import json
from datasets import Dataset
import os
from transformers import AutoTokenizer

PROCESSED_DATA_FOLDER = os.path.join("../data", "processed")
ANNOTATION_FILE = os.path.join(PROCESSED_DATA_FOLDER, "all.json")
OUTPUT_DATASET_FOLDER = os.path.join("../data", "ner_dataset")

MODEL_CHECKPOINT = "dmis-lab/biobert-base-cased-v1.1"
LABELS = ["CONDITION", "DRUG", "LAB_TEST", "VALUE", "OPERATOR", "PROCEDURE", "DEMOGRAPHIC"]

with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

dataset = Dataset.from_dict({
"text": [item["text"] for item in raw_data],
"ner_tags_spans": [item.get("label", []) for item in raw_data]
})
print(f"Loaded {len(dataset)} annotated examples from '{ANNOTATION_FILE}'.")

tag_names = ['O'] + [f'{prefix}-{tag}' for tag in LABELS for prefix in ['B', 'I']]
tag2id = {tag: i for i, tag in enumerate(tag_names)}
id2tag = {i: tag for tag, i in tag2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=False)
    all_labels = []
    for i, spans in enumerate(examples["ner_tags_spans"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = {} 
        for start, end, label in spans:
            for token_idx in range(len(tokenized_inputs.input_ids[i])):
                char_start, char_end = tokenized_inputs.token_to_chars(i, token_idx)
                if char_start is None: continue
                
                if max(start, char_start) < min(end, char_end):
                    word_id = word_ids[token_idx]
                    if word_id is not None:
                        prefix = "B"
                        if token_idx > 0 and word_ids[token_idx-1] == word_id:
                            if label_ids.get(word_id) is not None:
                                prefix = "I"
                        
                        tag = f"{prefix}-{label}"
                        if tag in tag2id:
                            label_ids[word_id] = tag2id[tag]
                        else:
                            print(f"Warning: Label '{label}' not found in schema. Skipping.")
        
        final_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                final_labels.append(-100)
            else:
                label_id = label_ids.get(word_idx)
                if label_id is not None:

                    if word_idx == previous_word_idx:
                        tag = tag_names[label_id].replace("B-", "I-")
                        final_labels.append(tag2id.get(tag, label_id))
                    else:
                        final_labels.append(label_id)
                else:
                    final_labels.append(tag2id['O'])
            previous_word_idx = word_idx
        all_labels.append(final_labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

print("\nTokenizing text and aligning labels with the IOB2 scheme...")
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset.column_names
)

final_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
print("\nSplitting data into a training set (90%) and an evaluation set (10%):")
print(final_dataset)

final_dataset.save_to_disk(OUTPUT_DATASET_FOLDER)
print(f"\nProcessed dataset saved to '{OUTPUT_DATASET_FOLDER}'. Ready for training.")