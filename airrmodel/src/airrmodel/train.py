from airrmodel.dataset import RepertoireDataset
from airrmodel.scalers import get_standard_scalers


def train(config, model_path, scalers_cache_path):
    scalers = get_standard_scalers(config, scalers_cache_path)
