from airrmodel.dataset import RepertoireDataset


def test_dataset():
    # Test initialization
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

    train_dataset = RepertoireDataset(config, mode="train")
    valid_dataset = RepertoireDataset(config, mode="valid")

    features_train_patient0 = train_dataset[0]
    features_valid_patient0 = valid_dataset[0]

    # Test features extraction
    print(features_train_patient0["text_features"])
    assert isinstance(features_train_patient0["text_features"][0][0], str)
    assert features_train_patient0["text_features"][0][0] != "", (
        "junction_aa field should ALWAYS be present"
    )
    assert isinstance(features_train_patient0["label"], bool)

    print(features_train_patient0)
