import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

BINARY_COLUMNS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

SERVICE_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
SERVICE_MAP = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}


# Preprocessor
class Preprocessor:
    def __init__(self, target="Churn"):
        self.target = target
        self.encoder = None
        self.feature_columns = None
        self.onehot_columns = []  # columns passed to the OneHotEncoder at fit time
        self.categorical_feature_columns = []

    # Helpers
    def find_missing(self, df, stage="raw data"):
        """Log and return the columns that contain missing values."""
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            logger.info(f"Missing values ({stage}): none")
        else:
            logger.info(f"Missing values ({stage}):\n{missing.to_string()}")

        return missing

    def remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Removed duplicates: {before - len(df)}")
        return df

    def clean(self, df):
        df = df.copy()
        self.find_missing(df)

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        self.find_missing(df, stage="after converting TotalCharges")

        # Drop customerID
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        df = self.remove_duplicates(df)

        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Removed rows with missing values: {before - len(df)}")

        return df

    def _encode_binary_and_service(self, X, track=False):
        for column in BINARY_COLUMNS:
            if column in X.columns:
                X[column] = X[column].map(BINARY_MAP)
                if track:
                    self.categorical_feature_columns.append(column)

        for column in SERVICE_COLUMNS:
            if column in X.columns:
                X[column] = X[column].map(SERVICE_MAP)
                if track:
                    self.categorical_feature_columns.append(column)

        return X

    # Fit and transform
    def preprocess(self, df):
        df = self.clean(df)

        y = df[self.target]
        X = df.drop(columns=[self.target])

        self.categorical_feature_columns = []

        X = self._encode_binary_and_service(X, track=True)
        self.onehot_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        if self.onehot_columns:
            encoded = self.encoder.fit_transform(X[self.onehot_columns])
            encoded_columns = self.encoder.get_feature_names_out(self.onehot_columns)
            encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=X.index)
            X = pd.concat([X.drop(columns=self.onehot_columns), encoded_df], axis=1)
            self.categorical_feature_columns.extend(encoded_columns.tolist())

        y = y.map({"Yes": 1, "No": 0})
        self.feature_columns = X.columns.tolist()

        logger.info("Preprocessing completed")
        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")
        logger.info(f"Categorical columns (not scaled): {self.categorical_feature_columns}")

        return X, y

    def fit_transform(self, df):
        return self.preprocess(df)

    def transform(self, df):
        df = df.copy()
        df = df.drop(columns=["customerID", self.target], errors="ignore")

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        df = self._encode_binary_and_service(df)

        if self.onehot_columns and self.encoder is not None:
            encoded = self.encoder.transform(df[self.onehot_columns])
            encoded_columns = self.encoder.get_feature_names_out(self.onehot_columns)
            encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)
            df = pd.concat([df.drop(columns=self.onehot_columns), encoded_df], axis=1)

        df = df.reindex(columns=self.feature_columns, fill_value=0)

        if df.isnull().any().any():
            logger.warning("Transformed data still contains NaN values")

        return df


# Feature engineering
class FeatureEngineer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scale_columns = []

    def fit_transform(self, X, y, categorical_columns=None):
        if categorical_columns is None:
            categorical_columns = []

        self.scale_columns = [col for col in X.columns if col not in categorical_columns]

        logger.info(f"Columns to scale ({len(self.scale_columns)}): {self.scale_columns}")
        logger.info(f"Columns not scaled ({len(categorical_columns)}): {categorical_columns}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Train shape: {X_train.shape}, test shape: {X_test.shape}")

        self.scaler.fit(X_train[self.scale_columns])

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[self.scale_columns] = self.scaler.transform(X_train[self.scale_columns])
        X_test_scaled[self.scale_columns] = self.scaler.transform(X_test[self.scale_columns])

        logger.info("Scaling completed (numeric columns only)")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform(self, X):
        X = X.copy()
        X[self.scale_columns] = self.scaler.transform(X[self.scale_columns])
        return X