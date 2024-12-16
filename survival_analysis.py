import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import matplotlib.pyplot as plt


class SurvivalAnalysis:
    def __init__(self, data, duration_col, event_col, frailty_col=None):
        """
        Initialize the SurvivalAnalysis class.

        Parameters:
        - data: DataFrame containing the dataset.
        - duration_col: Name of the column with survival times.
        - event_col: Name of the column with event occurrence.
        - frailty_col: Name of the column for clustering (optional).
        """
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col
        self.frailty_col = frailty_col
        self.selected_model = None

    def select_model(self, model_name, **kwargs):
        """Select a specific survival model to use for analysis."""
        if model_name == "Cox Proportional Hazards":
            self.selected_model = lambda train_data: self._fit_cox(train_data)
        elif model_name == "Frailty Model":
            if not self.frailty_col:
                raise ValueError("Frailty column must be specified for Frailty Model.")
            self.selected_model = lambda train_data: self._fit_frailty(train_data)
        elif model_name == "Weibull AFT Model":
            self.selected_model = lambda train_data: self._fit_weibull_aft(train_data)
        elif model_name == "Log-Normal AFT Model":
            self.selected_model = lambda train_data: self._fit_lognormal_aft(train_data)
        elif model_name == "Cox PH (Ridge)":
            self.selected_model = lambda train_data: self._fit_coxnet(
                train_data, l1_ratio=kwargs.get("l1_ratio", 0.0)
            )
        elif model_name == "Cox PH (Lasso)":
            self.selected_model = lambda train_data: self._fit_coxnet(
                train_data, l1_ratio=kwargs.get("l1_ratio", 1.0)
            )
        elif model_name == "Random Survival Forest":
            self.selected_model = lambda train_data: self._fit_rsf(train_data, **kwargs)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _fit_cox(self, train_data):
        model = CoxPHFitter()
        model.fit(train_data, duration_col=self.duration_col, event_col=self.event_col)
        return model

    def _fit_frailty(self, train_data):
        model = CoxPHFitter()
        model.fit(
            train_data,
            duration_col=self.duration_col,
            event_col=self.event_col,
            cluster_col=self.frailty_col,
        )
        return model

    def _fit_weibull_aft(self, train_data):
        model = WeibullAFTFitter()
        model.fit(train_data, duration_col=self.duration_col, event_col=self.event_col)
        return model

    def _fit_lognormal_aft(self, train_data):
        model = LogNormalAFTFitter()
        train_data = train_data.drop(columns=[self.frailty_col])
        model.fit(train_data, duration_col=self.duration_col, event_col=self.event_col)
        return model

    def _fit_coxnet(self, train_data, l1_ratio):
        X = train_data.drop(columns=[self.duration_col, self.event_col])
        y = train_data[[self.duration_col, self.event_col]].to_numpy()
        model = CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, l1_ratio=l1_ratio)
        model.fit(X, y)
        return model

    def _fit_rsf(self, train_data, **kwargs):
        X = train_data.drop(columns=[self.duration_col, self.event_col])
        y = np.array(
            [
                (row[self.event_col] == 1, row[self.duration_col])
                for _, row in train_data.iterrows()
            ],
            dtype=[("event", bool), ("time", float)],
        )
        model = RandomSurvivalForest(
            n_estimators=kwargs.get("n_estimators", 100),
            min_samples_split=kwargs.get("min_samples_split", 10),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            random_state=kwargs.get("random_state", 42),
        )
        model.fit(X, y)
        return model

    def cross_validate(self, n_splits=None):
        """
        Perform cross-validation with the selected model.

        Parameters:
        - n_splits: Number of folds for cross-validation.

        Returns:
        - A list with fold-wise C-index scores.
        """
        if not self.selected_model:
            raise ValueError(
                "No model has been selected. Use `select_model` to choose a model."
            )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        c_indices = []

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            model = self.selected_model(train_data)

            if isinstance(model, CoxPHFitter):
                predictions = model.predict_partial_hazard(test_data)
            elif isinstance(model, (WeibullAFTFitter, LogNormalAFTFitter)):
                predictions = model.predict_median(test_data)
            elif isinstance(model, RandomSurvivalForest):
                X_test = test_data.drop(columns=[self.duration_col, self.event_col])
                predictions = -model.predict(X_test)
            else:
                predictions = model.predict(
                    test_data.drop(columns=[self.duration_col, self.event_col])
                )

            c_index = concordance_index(
                test_data[self.duration_col], -predictions, test_data[self.event_col]
            )
            c_indices.append(c_index)

        return c_indices

    def compute_auc(self, time_points, n_splits=5):
        """
        Compute time-dependent AUC using cross-validation.

        Parameters:
        - time_points: List of time points to calculate AUC.
        - n_splits: Number of folds for cross-validation.

        Returns:
        - Mean AUC for each time point.
        """
        if not self.selected_model:
            raise ValueError(
                "No model has been selected. Use `select_model` to choose a model."
            )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_results = []

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            model = self.selected_model(train_data)

            if isinstance(model, RandomSurvivalForest):
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

                survival_functions = model.predict_survival_function(X_test)
                predictions = np.row_stack([
                    fn(time_points) for fn in survival_functions
                ])

                for time_point in time_points:
                    auc, _ = cumulative_dynamic_auc(
                        y_train, y_test, predictions[:, time_points.index(time_point)], time_point
                    )
                    auc_results.append((time_point, auc))

        return auc_results

    def plot_results(self, c_indices):
        """
        Plot a boxplot of the C-index scores for the selected model.

        Parameters:
        - c_indices: A list with fold-wise C-index scores.
        """
        plt.figure(figsize=(8, 5))
        plt.boxplot(c_indices, vert=True, patch_artist=True, labels=["C-index"])
        plt.ylabel("C-index")
        plt.title("Cross-Validation C-index Scores")
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

    # Initialize SurvivalAnalysis class
    sa = SurvivalAnalysis(data, duration_col, event_col, frailty_col)

    # Select and evaluate a model
    sa.select_model("Random Survival Forest", n_estimators=200, random_state=42)
    c_indices = sa.cross_validate()

    # Print results
    print("Cross-Validation C-index Scores:", c_indices)
    print("Average C-index:", np.mean(c_indices))

    # Compute AUC at specific time points
    time_points = [12, 24, 36]  # Example time points
    auc_results = sa.compute_auc(time_points)
    print("Time-dependent AUC:", auc_results)

    # Plot results
    sa.plot_results(c_indices)
