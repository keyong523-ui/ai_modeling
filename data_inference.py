import torch
import pandas as pd
import json
from tqdm import tqdm
from torch import nn
from transformers import AutoTokenizer, BertModel, BartForConditionalGeneration, PreTrainedTokenizerFast

# ------------------------------------------
# 0. Device
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------
# 1. Extractive Model 정의 (동일)
# ------------------------------------------
class BertForExtSummarization(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# ------------------------------------------
# 2. 모델 로드 (동일)
# ------------------------------------------
# Extractive
MODEL_EXT = "./kobert"
tok_ext = AutoTokenizer.from_pretrained(MODEL_EXT)

ext_model = BertForExtSummarization(MODEL_EXT)
ext_model.load_state_dict(torch.load("best_kobert_ext.pt", map_location=device))
ext_model.to(device)
ext_model.eval()

# Abstractive
MODEL_ABS = "./kobart"
tok_abs = PreTrainedTokenizerFast.from_pretrained(MODEL_ABS)

abs_model = BartForConditionalGeneration.from_pretrained(MODEL_ABS)
abs_model.load_state_dict(torch.load("best_kobart_abs.pt", map_location=device))
abs_model.to(device)
abs_model.eval()


# ------------------------------------------
# 3. 문장 분리
# ------------------------------------------
def split_sentences(text):
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 0]


# ------------------------------------------
# 4. Extractive top-3
# ------------------------------------------
def extractive_top3(sentences):
    if len(sentences) == 0:
        return []

    inputs = tok_ext(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = ext_model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )
        probs = torch.softmax(logits, dim=-1)[:, 1]

    probs = probs.cpu().tolist()

    ranked = sorted(
        [(sent, p) for sent, p in zip(sentences, probs)],
        key=lambda x: x[1],
        reverse=True
    )

    return [s for s, p in ranked[:3]]


# ------------------------------------------
# 5. Abstractive
# ------------------------------------------
def abstractive_summary(text):
    enc = tok_abs(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        generated_ids = abs_model.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_length=128,
            min_length=20,
            num_beams=8,
            no_repeat_ngram_size=2
        )

    return tok_abs.decode(generated_ids[0], skip_special_tokens=True)


# ------------------------------------------
# 6. CSV → 요약 생성
# ------------------------------------------
def summarize_csv(input_csv, output_json):
    df = pd.read_csv(input_csv)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        text = str(row["text"])

        # 1) 문장 분리
        sents = split_sentences(text)

        # 2) Extractive top-3
        ext3 = extractive_top3(sents)

        # 3) Abstractive
        abs_sum = abstractive_summary(text)

        # 결과 dict 구조
        results.append({
            "idx": row.get("idx", None),
            "title": row.get("title", None),
            "source": row.get("source", None),
            "published": row.get("published", None),
            "category": row.get("category", None),
            "text":row.get("text", None),
            "extractive_summary": ext3,
            "abstractive_summary": abs_sum
        })

    # JSON 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"JSON saved → {output_json}")


# ------------------------------------------
# 7. 실행
# ------------------------------------------
if __name__ == "__main__":
    summarize_csv(
        input_csv="news.csv",
        output_json="news.json"
    )
