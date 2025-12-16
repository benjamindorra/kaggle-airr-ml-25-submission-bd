import copy
import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

from airrmodel.dataset import RepertoireDataset


def to_categorical(df, categories_dict):
    """Converts a DataFrame column to categorical values based on provided categories."""
    df2 = df.copy()
    for feature, categories in categories_dict.items():
        df2[feature] = df2[feature].apply(lambda x: categories.index(x))
    return df2


def get_standard_scalers(config, cache_path):
    # Use cached standardization if available
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            categorical_scaler, numerical_scaler = pickle.load(f)
        return {"categorical": categorical_scaler, "numerical": numerical_scaler}

    assert config["validation_mode"] == "train_val_split", (
        "Only train_val_split mode is implemented"
    )

    bag_features_categories = {}
    numerical_features = {}
    # Load metadata
    dataset = RepertoireDataset(config, mode="train")
    metadata_df = dataset.metadata
    data_dir = dataset.data_dir
    # Standardize bag categorical features
    for bf in config["bag_categorical_features"]:
        bag_features_categories[bf] = metadata_df[bf].unique().tolist()
    categorical_df = to_categorical(metadata_df, bag_features_categories)
    # Standardize feature values
    categorical_scaler = StandardScaler()
    categorical_scaler.fit(categorical_df[config["bag_categorical_features"]])
    data_dfs = []
    for filename in metadata_df["filename"].tolist():
        data_dfs.append(pd.read_csv(os.path.join(data_dir, filename)))
    data_df = pd.concat(data_dfs)
    numerical_scaler = StandardScaler()
    numerical_scaler.fit(data_df[config["numerical_features"]])

    # Save scalers
    with open(cache_path, "wb") as f:
        pickle.dump((categorical_scaler, numerical_scaler), f)

    return {"categorical": categorical_scaler, "numerical": numerical_scaler}
