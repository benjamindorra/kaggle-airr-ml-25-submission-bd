import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class RawRepertoireDataset(Dataset):
    def __init__(self, config, mode="train"):
        """
        Get the required features without any preprocessing.
        Arguments:
            mode ("train", "valid" or "test") select the data to load
            config contain the model configuration parameters including the training/validation split
        """

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
            tf: data[tf].values.tolist() if tf in data.columns else [np.nan]
            for tf in self.text_features
        }

        numerical_features = {
            cf: data[cf].values.tolist() if cf in data.columns else [np.nan]
            for cf in self.numerical_features
        }

        # Load the patient-level features
        bag_categorical_features = {
            bcf: self.metadata.iloc[idx][bcf]
            if bcf in self.metadata.columns
            else np.nan
            for bcf in self.bag_categorical_features
        }

        bag_numerical_features = {
            bnf: self.metadata.iloc[idx][bnf]
            if bnf in self.metadata.columns
            else np.nan
            for bnf in self.bag_numerical_features
        }

        return {
            "text_features": text_features,
            "numerical_features": numerical_features,
            "bag_categorical_features": bag_categorical_features,
            "bag_numerical_features": bag_numerical_features,
            "label": self.labels[idx],
        }
