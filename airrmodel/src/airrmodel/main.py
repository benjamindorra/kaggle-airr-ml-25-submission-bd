import mlflow

from airrmodel.train import train


def get_config():
    """
    Configuration for the current run
    """
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
    return config


def main():
    config = get_config()
    model_path = "ems2-attention-mil.pt"
    scalers_cache_path = "scalers.pkl"
    train(config=config, model_path=model_path, scalers_cache_path=scalers_cache_path)
    # model = torch.jit.load(model_path)
    # test(model=model)


if __name__ == "__main__":
    main()
