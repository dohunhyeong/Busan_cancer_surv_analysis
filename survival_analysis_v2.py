import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import matplotlib.pyplot as plt

# Cox Proportional Hazards Model
class CoxPHModel:
    def __init__(self, duration_col, event_col):
        self.duration_col = duration_col
        self.event_col = event_col
        self.model = CoxPHFitter()

    def fit(self, train_data):
        self.model.fit(train_data, duration_col=self.duration_col, event_col=self.event_col)
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
        self.model.fit(train_data, duration_col=self.duration_col, event_col=self.event_col)
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
        self.model.fit(train_data, duration_col=self.duration_col, event_col=self.event_col)
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
                predictions
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
                predictions = np.row_stack([
                    fn(time_points) for fn in survival_functions
                ])

                for time_point in time_points:
                    auc, _ = cumulative_dynamic_auc(
                        y_train, y_test, predictions[:, time_points.index(time_point)], time_point
                    )
                    auc_results.append((time_point, auc))

        return auc_results

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


# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("C:/Users/user/Desktop/Survival_Data/DISEASE/colon.csv")
    duration_col = "stime"
    event_col = "event_inc"
    frailty_col = "gu_encoded"

    # Preprocess data
    data = data[
        [
            "sex",
            "tx_1",
            "tx_2",
            "tx_3",
            "tx_4",
            "tx_5",
            "seer_TF",
            "gu_encoded",
            "stime",
            "event_inc",
        ]
    ]
    data["stime"] = data["stime"].apply(lambda x: 0.01 if x <= 0 else x)

    # Cross-validation
    cv = CrossValidator(data, duration_col, event_col)

    # Evaluate Cox Proportional Hazards Model
    cox_c_indices = cv.cross_validate(CoxPHModel)
    print("Cox Proportional Hazards C-index:", np.mean(cox_c_indices))
    cv.plot_c_index(cox_c_indices, "Cox Proportional Hazards")

    # Evaluate Random Survival Forest Model
    rsf_c_indices = cv.cross_validate(RandomSurvivalForestModel, n_estimators=200, random_state=42)
    print("Random Survival Forest C-index:", np.mean(rsf_c_indices))
    cv.plot_c_index(rsf_c_indices, "Random Survival Forest")

    # Compute AUC for RSF
    time_points = [12, 24, 36]
    rsf_auc = cv.compute_auc(RandomSurvivalForestModel, time_points, n_splits=5, n_estimators=200, random_state=42)
    print("RSF Time-dependent AUC:", rsf_auc)
