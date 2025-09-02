import pandas as pd
import spacy


def segment_criteria(texts: list[str]) -> list[list[str]]:
    results = []
    try:
        nlp = spacy.load("en_core_sci_sm")
    except OSError:
        print("Error")
        return results

    for doc in nlp.pipe(texts, batch_size=100, n_process=4):
        segmented_criteria = []
        for sent in doc.sents:
            cleaned_sent = sent.text.strip()
            if len(cleaned_sent) > 10:
                segmented_criteria.append(cleaned_sent)
        results.append(segmented_criteria)

    return results


def segmentation(df):
    df["eligibility_criteria_clean"] = (
        df["eligibility_criteria"]
        .fillna("")
        .str.replace("•", "\n-")
        .str.replace("*", "\n-")
    )
    df["segmented_criteria"] = segment_criteria(df["eligibility_criteria_clean"].tolist())
    return df
