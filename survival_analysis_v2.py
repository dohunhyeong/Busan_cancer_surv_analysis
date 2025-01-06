import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

    def summary(self):
        return self.model.print_summary()


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

    def summary(self):
        return self.model.print_summary()


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

    def summary(self):
        return self.model.print_summary()


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
        self.model = CoxPHFitter(
            strata=strata_col, penalizer=penalizer, l1_ratio=l1_ratio
        )

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

    def summary(self):
        return self.model.print_summary()


# Cross-Validation Utility
class CrossValidator:
    def __init__(self, data, duration_col, event_col):
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col

    def cross_validate(self, model_class, n_splits=5, **model_kwargs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        c_indices = []

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            model = model_class(self.duration_col, self.event_col, **model_kwargs)
            model.fit(train_data)
            predictions = model.predict(test_data)

            c_index = concordance_index_censored(
                test_data[self.event_col] == 1,
                test_data[self.duration_col],
                predictions,
            )[0]
            c_indices.append(c_index)

        return c_indices

    def compute_auc(self, model_class, time_points, n_splits=5, **model_kwargs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_results = []

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            model = model_class(self.duration_col, self.event_col, **model_kwargs)
            model.fit(train_data)

            if isinstance(model, RandomSurvivalForestModel):
                X_train = train_data.drop(columns=[self.duration_col, self.event_col])
                X_test = test_data.drop(columns=[self.duration_col, self.event_col])
                y_train = np.array(
                    [
                        (row[self.event_col] == 1, row[self.duration_col])
                        for _, row in train_data.iterrows()
                    ],
                    dtype=[("event", bool), ("time", float)],
                )
                y_test = np.array(
                    [
                        (row[self.event_col] == 1, row[self.duration_col])
                        for _, row in test_data.iterrows()
                    ],
                    dtype=[("event", bool), ("time", float)],
                )

                survival_functions = model.model.predict_survival_function(X_test)
                predictions = np.row_stack(
                    [fn(time_points) for fn in survival_functions]
                )

                for time_point in time_points:
                    auc, _ = cumulative_dynamic_auc(
                        y_train,
                        y_test,
                        predictions[:, time_points.index(time_point)],
                        time_point,
                    )
                    auc_results.append((time_point, auc))

        return auc_results

    def plot_c_index(c_indices, model_name):
        """
        Plot a boxplot of the C-index scores for the selected model.

        Parameters:
        - c_indices: A list with fold-wise C-index scores.
        - model_name: The name of the model for labeling.
        """
        # Boxplot 생성
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")  # 전체 배경 하얀색
        box = ax.boxplot(
            c_indices,
            vert=True,
            patch_artist=True,  # 박스 색상 변경 가능
            labels=["C-index"],
            boxprops=dict(facecolor="lightblue", color="black"),  # 박스 색과 테두리
            whiskerprops=dict(color="black"),  # 수염 색상
            capprops=dict(color="black"),  # 끝 캡 색상
            medianprops=dict(color="yellow"),  # 중간선 색상
            flierprops=dict(
                markerfacecolor="red", markeredgecolor="black"
            ),  # 아웃라이어 스타일
        )

        # 텍스트 및 축 스타일 설정
        ax.set_ylabel("C-index", color="black")
        ax.set_title(f"Cross-Validation C-index Scores for {model_name}", color="black")
        ax.tick_params(colors="black")  # x축, y축 눈금 색상

        # 격자 스타일 설정
        ax.grid(axis="y", linestyle="--", color="lightgray", alpha=0.7)

        # 레이아웃 및 표시
        plt.tight_layout()
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


# Example usage
# if __name__ == "__main__":
#     # Load data
#     data = pd.read_csv("C:/Users/user/Desktop/Survival_Data/DISEASE/colon.csv")
#     duration_col = "stime"
#     event_col = "event_inc"
#     frailty_col = "gu_encoded"

#     # Preprocess data
#     data = data[
#         [
#             "sex",
#             "tx_1",
#             "tx_2",
#             "tx_3",
#             "tx_4",
#             "tx_5",
#             "seer_TF",
#             "gu_encoded",
#             "stime",
#             "event_inc",
#         ]
#     ]
#     data["stime"] = data["stime"].apply(lambda x: 0.01 if x <= 0 else x)

#     # Cross-validation
#     cv = CrossValidator(data, duration_col, event_col)

#     # Evaluate Cox Proportional Hazards Model
#     cox_c_indices = cv.cross_validate(CoxPHModel)
#     print("Cox Proportional Hazards C-index:", np.mean(cox_c_indices))
#     cv.plot_c_index(cox_c_indices, "Cox Proportional Hazards")

#     # Evaluate Random Survival Forest Model
#     rsf_c_indices = cv.cross_validate(RandomSurvivalForestModel, n_estimators=200, random_state=42)
#     print("Random Survival Forest C-index:", np.mean(rsf_c_indices))
#     cv.plot_c_index(rsf_c_indices, "Random Survival Forest")

#     # Compute AUC for RSF
#     time_points = [12, 24, 36]
#     rsf_auc = cv.compute_auc(RandomSurvivalForestModel, time_points, n_splits=5, n_estimators=200, random_state=42)
#     print("RSF Time-dependent AUC:", rsf_auc)
