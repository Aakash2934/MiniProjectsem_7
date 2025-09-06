import re
from typing import List
from tqdm import tqdm

def segment_text_batch(texts: List[str], nlp_model) -> List[List[str]]:
    final_results = []
    for doc in tqdm(nlp_model.pipe(texts, batch_size=50), total=len(texts), desc="Segmenting criteria"):
        cleaned_lines = []
        text_block = str(doc).replace('•', '\n-').replace('*', '\n-')
        for line in text_block.split('\n'):
            line = line.strip()
            if not line:
                continue
            cleaned_line = re.sub(r'^\s*-\s*|\d+\.\s*', '', line).strip()
            cleaned_lines.append(cleaned_line)
        cleaned_doc = nlp_model(" ".join(cleaned_lines))
        sentences = [sent.text.strip() for sent in cleaned_doc.sents if len(sent.text.strip()) > 10]
        final_results.append(sentences)
        
    return final_results

