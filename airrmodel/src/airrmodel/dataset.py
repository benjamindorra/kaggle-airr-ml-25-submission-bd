import os

import pandas as pd
from torch.utils.data import Dataset


class RepertoireDataset(Dataset):
    def __init__(self, config, mode="train"):
        def load_metadata(data_dir, datasets):
            metadata_dfs = []
            for dataset in datasets:
                df = pd.read_csv(os.path.join(data_dir, dataset, "metadata.csv"))
                df["filename"] = df["filename"].apply(
                    lambda x: os.path.join(dataset, x)
                )
                metadata_dfs.append(df)
            return pd.concat(metadata_dfs)

        self.data_dir = config["data_dir"]
        # Load all metadata csv files
        #  convert paths to individual data points into paths relative to the data_dir
        # concatenate the dataframes
        if mode == "train":
            self.metadata = load_metadata(self.data_dir, config["training_datasets"])
        else:
            self.metadata = load_metadata(self.data_dir, config["validation_datasets"])

        self.filenames = self.metadata["filename"].tolist()
        self.labels = self.metadata["label_positive"].tolist()
        self.text_features = config["text_features"]
        self.numerical_features = config["numerical_features"]
        self.bag_categorical_features = config["bag_categorical_features"]
        self.bag_numerical_features = config["bag_numerical_features"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = pd.read_csv(os.path.join(self.data_dir, filename), sep="\t")

        # Load the available features or use an empty token for unavailable ones
        text_features = {
            tf: data[tf].values.tolist()
            for tf in self.text_features
            if tf in data.columns
        }

        numerical_features = {
            cf: data[cf].values.tolist()
            for cf in self.numerical_features
            if cf in data.columns
        }

        bag_categorical_features = {
            bcf: self.metadata.iloc[idx][bcf]
            for bcf in self.bag_categorical_features
            if bcf in self.metadata.columns
        }

        bag_numerical_features = {
            bnf: self.metadata.iloc[idx][bnf]
            for bnf in self.bag_numerical_features
            if bnf in self.metadata.columns
        }

        return {
            "text_features": text_features,
            "numerical_features": numerical_features,
            "bag_categorical_features": bag_categorical_features,
            "bag_numerical_features": bag_numerical_features,
            "label": self.labels[idx],
        }
