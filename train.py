import os
import pandas as pd
import json
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm.auto import tqdm
from datasets import load_metric
from datetime import datetime

# W&B 로그인
wandb.login(key ='a8fa41ae061743cfb789129f6a363a02ff8ecebc')

# 전처리 함수
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^1-9a-zA-Z가-힣?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

# 전역 변수 설정
seed = 42

# 데이터 불러오기
train_data_path = "./data/train.csv"
train_data = pd.read_csv(train_data_path)

label_encode = {
    "협박 대화": 0,
    "갈취 대화": 1,
    "직장 내 괴롭힘 대화": 2,
    "기타 괴롭힘 대화": 3,
}

intimidation = train_data[train_data['class'] == '협박 대화']
extortion = train_data[train_data['class'] == '갈취 대화']
harassment_workplace = train_data[train_data['class'] == '직장 내 괴롭힘 대화']
harassment_others = train_data[train_data['class'] == '기타 괴롭힘 대화']

# 통합 및 라벨링
concat_tr_df = pd.concat([intimidation, extortion, harassment_workplace, harassment_others], axis=0, ignore_index=True)
concat_tr_df['encoded_label'] = concat_tr_df['class'].map(label_encode)

train_texts = concat_tr_df['conversation'].to_list()
train_labels = concat_tr_df['encoded_label'].to_list()

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
# 실험 셋 로드
exp_set = pd.read_csv('./data/exp_set.csv')

# 오늘 날짜를 yymmdd 형식으로 변환
today_date = datetime.now().strftime('%y%m%d')
i = 0

for idx, row in exp_set.iterrows():
    print(idx)
    i += 1
    run_nm = today_date + f"_{i}"
    
    preprocessing_function = globals()[row['preprocessing']]
    test_size = row['test_size']
    model_name = row['model_name']
    learning_rate = row['learning_rate']
    epochs = row['epochs']
    batch_size = row['batch_size']

    # 모델 구성 및 하이퍼파라미터 로깅
    wandb.config = {
        'preprocessing': row['preprocessing'],
        'test_size': test_size,
        'model_name': model_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }
    
    if preprocessing_function:
        train_texts = [preprocessing_function(sentence) for sentence in train_texts]
    
    # 데이터 전처리
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=test_size, random_state=seed, stratify=train_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encode))

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    # wandb 초기화
    wandb.init(project="test2", entity="dogcat1943", name=f"{run_nm}", config=wandb.config)

    # 학습 루프
    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(100)

            # wandb에 학습 손실 기록
            wandb.log({"train_loss": loss.item()})

    # 평가
    metric = load_metric("f1", trust_remote_code=True)

    model.eval()
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    final_score = metric.compute(average='weighted')
    print(final_score)

    # wandb에 평가 메트릭 기록
    wandb.log({"f1_score": final_score["f1"]})

    # wandb 종료
    wandb.finish()