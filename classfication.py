import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from transformers import ElectraTokenizer, ElectraConfig, ElectraForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

# ============================================
# 1) DATA LOAD
# ============================================
train_df = pd.read_csv("data/train_sentence.csv")
valid_df = pd.read_csv("data/val_sentence.csv")

label2id = {"사실형": 0, "추론형": 1, "예측형": 2}
id2label = {v: k for k, v in label2id.items()}

train_df["label_id"] = train_df["label"].map(label2id)
valid_df["label_id"] = valid_df["label"].map(label2id)

# ============================================
# 2) Dataset
# ============================================

class SentenceTypeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        text = self.df.loc[idx, "sentence"]
        label = self.df.loc[idx, "label_id"]

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length"
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ============================================
# 3) Tokenizer & Loader
# ============================================

model_name = "./koelectra"
tokenizer = ElectraTokenizer.from_pretrained(model_name)

train_dataset = SentenceTypeDataset(train_df, tokenizer)
valid_dataset = SentenceTypeDataset(valid_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)


# ============================================
# 4) Model Config Override (num_labels=3 강제 적용)
# ============================================

config = ElectraConfig.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

model = ElectraForSequenceClassification.from_pretrained(
    model_name,
    config=config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============================================
# 5) Optimizer / Scheduler
# ============================================

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 20

total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# ============================================
# 6) Train & Validate
# ============================================

def validate_epoch(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(labels_all, preds_all, average="macro")

    return avg_loss, f1, preds_all, labels_all


best_f1 = -1
save_path = "./best_sentence_type_model"

for epoch in range(num_epochs):
    print(f"\n===== EPOCH {epoch+1} =====")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # ------- VALIDATION (F1 기반 BEST SAVE) -------
    val_loss, val_f1, preds_all, labels_all = validate_epoch(model, valid_loader, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Valid Loss: {val_loss:.4f} | Valid F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        print(f"🔥 Best model updated! F1 {best_f1:.4f} → {val_f1:.4f}")
        best_f1 = val_f1

        if os.path.exists(save_path):
            os.system(f"rm -rf {save_path}")

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


# ============================================
# 7) 최종 Best 모델 불러와서 평가 결과 출력
# ============================================

print("\n===== FINAL VALIDATION EVALUATION =====")
best_model = ElectraForSequenceClassification.from_pretrained(save_path).to(device)
best_model.eval()

_, _, preds_all, labels_all = validate_epoch(best_model, valid_loader, device)

print("\nClassification Report:")
print(classification_report(labels_all, preds_all, target_names=label2id.keys()))

print("Confusion Matrix:")
print(confusion_matrix(labels_all, preds_all))

print(f"\nBest F1 Score: {best_f1:.4f}")
