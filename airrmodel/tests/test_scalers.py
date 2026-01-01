import os

from typing_extensions import Dict

from airrmodel.config import get_config
from airrmodel.scalers import get_standard_scalers


def test_get_scalers():
    config = get_config()
    # Reduce dataset size to accelerate test
    config["training_datasets"] = ["train_dataset_8"]

    # test initialization
    print(os.getcwd())
    cache_path = "test_cache"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    scalers = get_standard_scalers(config, cache_path)

    assert len(scalers) == 3

    assert all(isinstance(scaler, Dict) for scaler in scalers.values())

    # test caching
    scalers = get_standard_scalers(config, cache_path)

    assert len(scalers) == 3

    assert all(isinstance(scaler, Dict) for scaler in scalers.values())

    print(scalers)
