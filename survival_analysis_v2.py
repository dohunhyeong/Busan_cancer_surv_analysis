import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sksurv.metrics import brier_score as sk_brier_score

# Cox Proportional Hazards Model
class CoxPHModel:
    def __init__(self, duration_col, event_col, penalizer=0.0, l1_ratio=0.0):
        self.duration_col = duration_col
        self.event_col = event_col
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    def fit(self, train_data):
        self.model.fit(
            train_data, duration_col=self.duration_col, event_col=self.event_col
        )
        return self

    def predict(self, test_data):
        return self.model.predict_partial_hazard(test_data)


# Weibull Accelerated Failure Time Model
class WeibullAFTModel:
    def __init__(self, duration_col, event_col):
        self.duration_col = duration_col
        self.event_col = event_col
        self.model = WeibullAFTFitter()

    def fit(self, train_data):
        self.model.fit(
            train_data, duration_col=self.duration_col, event_col=self.event_col
        )
        return self

    def predict(self, test_data):
        return self.model.predict_median(test_data)


# Log-Normal Accelerated Failure Time Model
class LogNormalAFTModel:
    def __init__(self, duration_col, event_col):
        self.duration_col = duration_col
        self.event_col = event_col
        self.model = LogNormalAFTFitter()

    def fit(self, train_data):
        self.model.fit(
            train_data, duration_col=self.duration_col, event_col=self.event_col
        )
        return self

    def predict(self, test_data):
        return self.model.predict_median(test_data)


# Random Survival Forest Model
class RandomSurvivalForestModel:
    def __init__(self, duration_col, event_col, **kwargs):
        self.duration_col = duration_col
        self.event_col = event_col
        self.model = RandomSurvivalForest(**kwargs)

    def fit(self, train_data):
        X = train_data.drop(columns=[self.duration_col, self.event_col])
        y = np.array(
            [
                (row[self.event_col] == 1, row[self.duration_col])
                for _, row in train_data.iterrows()
            ],
            dtype=[("event", bool), ("time", float)],
        )
        self.model.fit(X, y)
        return self

    def predict(self, test_data):
        X = test_data.drop(columns=[self.duration_col, self.event_col])
        return -self.model.predict(X)


# Frailty Model
class Frailty:
    def __init__(
        self, duration_col, event_col, strata_col, penalizer=0.0, l1_ratio=0.0
    ):
        self.duration_col = duration_col
        self.event_col = event_col
        self.strata_col = strata_col
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    def fit(self, train_data):
        self.model.fit(
            train_data,
            duration_col=self.duration_col,
            event_col=self.event_col,
            strata=self.strata_col,
        )
        return self

    def predict(self, test_data):
        return self.model.predict_partial_hazard(test_data)


# CrossValidator class
class CrossValidator:
    def __init__(self, data, duration_col, event_col):
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col

    def train_model(self, model_class, train_data, **model_kwargs):
        # 학습 데이터를 사용하여 모델 학습
        model = model_class(self.duration_col, self.event_col, **model_kwargs)
        model.fit(train_data)
        return model

    def evaluate_model(self, model, test_data):
        # 테스트 데이터를 사용하여 C-index 계산
        predictions = model.predict(test_data)
        c_index = concordance_index_censored(
            test_data[self.event_col] == 1,
            test_data[self.duration_col],
            predictions,
        )[0]
        return c_index



    def calculate_brier_score(self, model, test_data, times):
        # times가 이미 float으로 변환되어 있음

        # 테스트 데이터를 구조화된 배열로 변환
        y_test_structured = np.array(
            [
                (row[self.event_col] == 1, row[self.duration_col])
                for _, row in test_data.iterrows()
            ],
            dtype=[("event", bool), ("time", float)],
        )

        # 생존 확률 계산 (생존 확률 함수 반환)
        survival_functions = model.model.predict_survival_function(
            test_data.drop(columns=[self.duration_col, self.event_col])
        )

        # 각 시간점에 대해 생존 확률을 2D 배열로 변환
        survival_probs = np.row_stack([fn(times) for fn in survival_functions])

        # Brier Score 계산
        brier_scores = sk_brier_score(y_test_structured, survival_probs, times)
        return brier_scores



    # 수정된 cross_validate 메서드 내에서 float 변환
    def cross_validate(self, model_class, n_splits=5, **model_kwargs):
        # 전체 교차 검증 프로세스
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        c_indices = []
        brier_scores_list = []

        # 공통 시간점 생성 (float으로 변환)
        all_times = np.linspace(1, float(self.data[self.duration_col].max()), 50)

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            # 1. 모델 학습
            model = self.train_model(model_class, train_data, **model_kwargs)

            # 2. C-index 계산
            c_index = self.evaluate_model(model, test_data)
            c_indices.append(c_index)

            # 3. Brier Score 계산
            brier_scores = self.calculate_brier_score(model, test_data, all_times)
            brier_scores_list.append(brier_scores)

        return c_indices, brier_scores_list



    def plot_c_index(self, c_indices, model_name):
        """
        Plot a boxplot of the C-index scores for the selected model.

        Parameters:
        - c_indices: A list with fold-wise C-index scores.
        - model_name: The name of the model for labeling.
        """
        plt.figure(figsize=(8, 5))
        plt.boxplot(c_indices, vert=True, patch_artist=True, labels=["C-index"])
        plt.ylabel("C-index")
        plt.title(f"Cross-Validation C-index Scores for {model_name}")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def plot_brier_scores(self, brier_scores_list, times, model_name):
        """
        Plot the average Brier scores over time for the selected model.

        Parameters:
        - brier_scores_list: A list of Brier scores for each fold.
        - times: Array of time points.
        - model_name: The name of the model for labeling.
        """
        mean_brier_scores = np.mean(brier_scores_list, axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(times, mean_brier_scores, label=f"{model_name} Brier Score")
        plt.title(f"Brier Score over Time ({model_name})")
        plt.xlabel("Time")
        plt.ylabel("Brier Score")
        plt.legend()
        plt.grid()
        plt.show()



# DeepSurv 모델 정의
class DeepSurv(nn.Module):
    def __init__(self, input_dim, activation_fn=nn.ReLU):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation_fn = activation_fn()

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        return self.fc3(x)


# Dataset 정의
class SurvivalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.durations = torch.tensor(y[:, 0], dtype=torch.float32)
        self.events = torch.tensor(y[:, 1], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]
