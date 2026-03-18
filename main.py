import torch, torch.nn as nn
from transformers import BertTokenizerFast, BertForTokenClassification
 
LABELS = {'O':0,'B-PROBLEM':1,'I-PROBLEM':2,'B-TEST':3,'I-TEST':4,'B-TREATMENT':5,'I-TREATMENT':6}
ID2LABEL = {v:k for k,v in LABELS.items()}
 
class ClinicalNER(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            'bert-base-uncased', num_labels=len(LABELS))
        # CRF-like output (simplified)
        self.crf_transitions = nn.Parameter(torch.randn(len(LABELS), len(LABELS)))
 
    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids, attention_mask=attention_mask, labels=labels)
 
def viterbi_decode(emissions, transitions):
    """Viterbi algorithm for CRF decoding."""
    n_tags = emissions.shape[1]
    T = emissions.shape[0]
    viterbi = emissions[0].clone()
    backpointers = []
    for t in range(1, T):
        trans_score = viterbi.unsqueeze(1) + transitions
        best_scores, best_tags = trans_score.max(0)
        viterbi = best_scores + emissions[t]
        backpointers.append(best_tags)
    best_path = [viterbi.argmax().item()]
    for bp in reversed(backpointers):
        best_path.insert(0, bp[best_path[0]].item())
    return best_path
 
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = ClinicalNER()
 
clinical_texts = [
    "Patient presents with chest pain and shortness of breath.",
    "Administered 500mg aspirin and ordered ECG test.",
    "Diagnosed with acute myocardial infarction, treatment started.",
]
for text in clinical_texts:
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors='pt',
                    truncation=True, padding='max_length', max_length=64)
    with torch.no_grad():
        out = model(enc['input_ids'], enc['attention_mask'])
    word_ids = enc.word_ids()
    preds = out.logits.argmax(-1)[0]
    result = []
    for wid, pred in zip(word_ids, preds):
        if wid is not None and (not result or result[-1][0] != wid):
            result.append((wid, ID2LABEL[pred.item()]))
           for wid, label in result:
        if label != 'O': print(f"  '{words[wid]}' → {label}")
