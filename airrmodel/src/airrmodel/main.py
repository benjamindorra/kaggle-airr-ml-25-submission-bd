import mlflow

from airrmodel.config import get_config
from airrmodel.train import train


def main():
    config = get_config()
    model_path = "ems2-attention-mil.pt"
    scalers_cache_path = "scalers.pkl"
    train(config=config, model_path=model_path, scalers_cache_path=scalers_cache_path)
    # model = torch.jit.load(model_path)
    # test(model=model)


if __name__ == "__main__":
    main()
