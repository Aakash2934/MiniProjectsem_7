import re
from typing import List

def segment_text_batch(texts: List[str], nlp_model) -> List[List[str]]:
    cleaned_texts = []

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            cleaned_texts.append("")
            continue

        text = text.replace('•', '\n-').replace('*', '\n-')
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            cleaned_line = re.sub(r'^\s*-\s*|\d+\.\s*', '', line).strip()
            cleaned_lines.append(cleaned_line)
        cleaned_texts.append("\n".join(cleaned_lines))
    final_results = []
    for doc in nlp_model.pipe(cleaned_texts, batch_size=100, n_process=-1):
        sentences = [
            sent.text.strip() for sent in doc.sents 
            if len(sent.text.strip()) > 10
        ]
        final_results.append(sentences)
        
    return final_results

