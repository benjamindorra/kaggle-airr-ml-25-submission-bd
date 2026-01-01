import copy
import os
import pickle
from math import isnan

import numpy as np
import pandas as pd

from airrmodel.rawdataset import RawRepertoireDataset


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
            scalers = pickle.load(f)
        return scalers

    assert config["validation_mode"] == "train_val_split", (
        "Only train_val_split mode is implemented"
    )

    numerical_scales = {
        k: {"mean": 0.0, "std": 1.0, "counter": 0} for k in config["numerical_features"]
    }
    bag_categorical_matching = {k: set() for k in config["bag_categorical_features"]}
    bag_numerical_scales = {
        k: {"mean": 0.0, "std": 1.0, "counter": 0}
        for k in config["bag_numerical_features"]
    }
    # Load metadata
    dataset = RawRepertoireDataset(config, mode="train")
    # Accumulate mean, std, and categories
    for features in dataset:
        for f in features["numerical_features"]:
            for v in features["numerical_features"][f]:
                if not isnan(v):
                    numerical_scales[f]["mean"] += v
                    numerical_scales[f]["std"] += v**2
                    numerical_scales[f]["counter"] += 1
        for f in features["bag_categorical_features"]:
            bag_categorical_matching[f].add(features["bag_categorical_features"][f])
        for f in features["bag_numerical_features"]:
            bag_numerical_scales[f]["mean"] += features["bag_numerical_features"][f]
            bag_numerical_scales[f]["std"] += features["bag_numerical_features"][f] ** 2
            bag_numerical_scales[f]["counter"] += 1

    # Get final standardization values assuming gaussian distributions
    for f in numerical_scales:
        numerical_scales[f]["mean"] /= numerical_scales[f]["counter"]
        numerical_scales[f]["std"] /= numerical_scales[f]["counter"]
        numerical_scales[f]["std"] -= numerical_scales[f]["mean"] ** 2
        numerical_scales[f]["std"] = numerical_scales[f]["std"] ** 0.5

    for f in bag_numerical_scales:
        bag_numerical_scales[f]["mean"] /= bag_numerical_scales[f]["counter"]
        bag_numerical_scales[f]["std"] /= bag_numerical_scales[f]["counter"]
        bag_numerical_scales[f]["std"] -= bag_numerical_scales[f]["mean"] ** 2
        bag_numerical_scales[f]["std"] = bag_numerical_scales[f]["std"] ** 0.5

    # Convert categorical features to standardized numerical values
    for f in bag_categorical_matching:
        # Convert to number, ignoring nan values
        categories = list(bag_categorical_matching[f])
        print(categories)
        matching = {
            k: i
            for i, k in enumerate(categories)
            if (isinstance(k, str)) or (not isnan(k))
        }
        # Standardize
        mean = np.mean(list(matching.values()))
        std = np.std(list(matching.values()))
        bag_categorical_matching[f] = {k: (v - mean) / std for k, v in matching.items()}

    scalers = {
        "numerical_scales": numerical_scales,
        "bag_categorical_matching": bag_categorical_matching,
        "bag_numerical_scales": bag_numerical_scales,
    }
    # Save matchings and scales for later use
    with open(cache_path, "wb") as f:
        pickle.dump(scalers, f)

    return scalers
