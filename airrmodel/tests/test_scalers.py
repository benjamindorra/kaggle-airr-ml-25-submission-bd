import os

from airrmodel.scalers import get_standard_scalers


def test_get_scalers():
    config = {
        "validation_mode": "train_val_split",
        "data_dir": "../train_datasets/train_datasets",
        "validation_datasets": ["train_dataset_1"],
        "training_datasets": [
            "train_dataset_2",
            "train_dataset_3",
            "train_dataset_4",
            "train_dataset_5",
            "train_dataset_6",
            "train_dataset_7",
            "train_dataset_8",
        ],
        "epochs": 1,
        "features_extractor": "esm2_t6_8M_UR50D",
        "features_dropout": 0.5,
        "attention_dim": 64,
        "num_classes": 1,
        "num_features": 5,
        "text_features": ["junction_aa", "v_call", "j_call", "d_call"],
        "numerical_features": ["templates"],
        "bag_categorical_features": [
            "study_group_description",
            "sex",
            "race",
            "A",
            "B",
            "C",
            "DPA1",
            "DPB1",
            "DQA1",
            "DQB1",
            "DRB1",
            "DRB3",
            "DRB4",
            "DRB5",
        ],
        "bag_numerical_features": [
            "age",
        ],
    }
    # test initialization
    print(os.getcwd())
    cache_path = "test_cache"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    scalers = get_standard_scalers(config, cache_path)

    assert len(scalers) == 2

    assert all(isinstance(scaler, StandardScaler) for scaler in scalers.values())

    # test caching
    scalers = get_standard_scalers(config, cache_path)

    assert len(scalers) == 2

    assert all(isinstance(scaler, StandardScaler) for scaler in scalers.values())

    print(scalers)
