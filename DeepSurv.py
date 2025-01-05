from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import survival_analysis_v2 as sa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sksurv.metrics import concordance_index_censored

# Load data
data = pd.read_csv("C:/Users/user/Desktop/survival_data/DISEASE/colon.csv", index_col=0)

# Load data
data = pd.read_csv("C:/Users/user/Desktop/survival_data/DISEASE/colon.csv", index_col=0)
duration_col = "stime"
event_col = "event_inc"

# Preprocess data
data = data[
    [
        "stime",
        "event_inc",
        "tx_1",
        "tx_2",
        "tx_3",
        "tx_4",
        "tx_5",
        "seer_TF",
        "gu_encoded",
    ]
]
data["stime"] = data["stime"].apply(lambda x: 0.01 if x <= 0 else x)

# Preprocess data
data = data[
    [
        "stime",
        "event_inc",
        "tx_1",
        "tx_2",
        "tx_3",
        "tx_4",
        "tx_5",
        "seer_TF",
        "gu_encoded",
    ]
]
data["stime"] = data["stime"].apply(lambda x: 0.01 if x <= 0 else x)

# 데이터 분리
X = data.drop(columns=[duration_col, event_col]).values
y = data[[duration_col, event_col]].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PyTorch 데이터셋 생성
train_dataset = sa.SurvivalDataset(X_train, y_train)
test_dataset = sa.SurvivalDataset(X_test, y_test)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# DeepSurv 모델 초기화
input_dim = X_train.shape[1]
deepsurv_model = sa.DeepSurv(input_dim=input_dim, activation_fn=nn.ReLU)

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()  # 예시, 생존 분석에서는 Cox loss 함수가 일반적임
optimizer = torch.optim.Adam(deepsurv_model.parameters(), lr=0.001)


# 학습
# Training 함수 정의
def train_deepsurv(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, durations, events in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, events)  # 사용자 정의 Cox loss 함수 적용 필요
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")


# 모델 학습
train_deepsurv(deepsurv_model, train_loader, criterion, optimizer)


# 테스트 데이터에 대한 C-index 계산
def evaluate_deepsurv(model, test_loader, y_test):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _, _ in test_loader:
            outputs = model(X_batch).squeeze().numpy()
            predictions.extend(outputs)

    # C-index 계산
    c_index = concordance_index_censored(
        y_test[:, 1].astype(bool),  # 이벤트 발생 여부
        y_test[:, 0],  # 생존 시간
        predictions,
    )[0]
    return c_index


# 모델 평가
c_index_deepsurv = evaluate_deepsurv(deepsurv_model, test_loader, y_test)
print(f"DeepSurv Model C-index: {c_index_deepsurv}")
