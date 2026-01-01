from airrmodel.config import get_config
from airrmodel.dataset import RepertoireDataset


def test_dataset():
    # Test initialization
    config = get_config()
    # Reduce dataset size to accelerate test
    config["training_datasets"] = ["train_dataset_8"]

    train_dataset = RepertoireDataset(config, mode="train")
    valid_dataset = RepertoireDataset(config, mode="valid")

    features_train_patient0 = train_dataset[0]
    features_valid_patient0 = valid_dataset[0]

    # Test features extraction
    print(features_train_patient0["text_features"])
    assert isinstance(features_train_patient0["text_features"]["junction_aa"][0], str)
    assert features_train_patient0["text_features"]["junction_aa"][0] != "", (
        "junction_aa field should ALWAYS be present"
    )
    assert isinstance(features_train_patient0["label"], bool)

    print(features_train_patient0)
