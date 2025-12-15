import os

import pandas as pd


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
            self.metadata = load_metadata(self.data_dir, config["train_datasets"])
        else:
            self.metadata = load_metadata(self.data_dir, config["validation_datasets"])

        pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))
        self.filenames = self.metadata["filename"].tolist()
        self.labels = self.metadata["label_positive"].tolist()
        self.text_features = config["text_features"]
        self.categorical_features = config["categorical_features"]
        self.bag_categorical_features = config["bag_categorical_features"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = pd.read_csv(os.path.join(self.data_dir, filename))

        text_features = [
            data[tf].fillna("").values.tolist() for tf in self.text_features
        ]
        categorical_features = [
            data[cf].fillna("").values.tolist() for cf in self.categorical_features
        ]
        bag_categorical_features = [
            self.metadata.iloc[idx][bcf].fillna("").values.tolist()
            for bcf in self.bag_categorical_features
        ]

        return {
            "text_features": text_features,
            "categorical_features": categorical_features,
            "bag_categorical_features": bag_categorical_features,
            "labels": self.labels[idx],
        }
