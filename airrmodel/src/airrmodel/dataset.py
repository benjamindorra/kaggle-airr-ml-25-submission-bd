import os
from math import isnan

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EsmModel

from airrmodel.scalers import get_standard_scalers


class RepertoireDataset(Dataset):
    def __init__(
        self,
        config,
        mode="train",
        cache_extension="_embeddings.tsv",
    ):
        """
        Get the required features with preprocessing.
        Arguments:
            mode ("train", "valid" or "test") select the data to load
            config contain the model configuration parameters including the training/validation split
            cache_extension indicates the path to store the preprocessed features for faster use during training/validation
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
        self.tokenizer = AutoTokenizer.from_pretrained(config["features_extractor"])
        self.feature_extractor = EsmModel.from_pretrained(
            config["features_extractor"], add_pooling_layer=False
        )
        self.embed_dim = self.feature_extractor.config.hidden_size
        self.cache_extension = cache_extension
        self.scalers = get_standard_scalers(config, "train_scalers.pkl")
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        self.feature_extractor.to(self.device)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # If the data have already been preprocessed, load the cached preprocessed version
        cache_path = os.path.join(
            self.data_dir, os.path.splitext(filename)[0] + self.cache_extension
        )
        cached = False
        if os.path.exists(cache_path):
            data = pd.read_csv(cache_path, sep="\t")
            # if len(data) >= idx + 1:
            cached = True

        if not cached:
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

        # Preprocess antibody-level features
        if not cached:
            processed_text_features = {}
            for k, v in text_features.items():
                # Gradually accumulate tokens to limit the passes to the GPU and improve speed
                counter = 0
                partial_counter = 0
                batch_size = 128
                texts = []
                missing = []
                processed_text_features[k] = []
                for f in v:
                    if isinstance(f, str):
                        counter += 1
                        texts.append(f)
                    else:
                        missing.append(partial_counter)

                    if (counter % batch_size == 0) or counter >= len(v):
                        with torch.no_grad():
                            tokenized = self.tokenizer(
                                texts,
                                padding="longest",
                                truncation=True,
                                return_tensors="pt",
                            ).to(self.device)
                            features = self.feature_extractor(
                                tokenized.input_ids, tokenized.attention_mask
                            ).last_hidden_state.mean(axis=1)
                            features_list = list(torch.split(features, 1, dim=1))
                        # Add features and placeholders in the right order
                        for i in range(partial_counter):
                            if i in missing:
                                processed_text_features[k].append(
                                    torch.zeros([1, self.embed_dim])
                                )
                            else:
                                processed_text_features[k].append(features_list.pop(0))

                        processed_text_features[k].extend(
                            torch.split(features, 1, dim=0)
                        )
                        texts = []
                        missing = []
                        partial_counter = 0
                    partial_counter += 1
            text_features = processed_text_features

            print(text_features)

            numerical_features = {
                k: [
                    (f - self.scalers["numerical_scales"][k]["mean"])
                    / self.scalers["numerical_scales"][k]["std"]
                    if not isnan(f)
                    else 0.0
                    for f in v
                ]
                for k, v in numerical_features.items()
            }

            # Save the preprocessed features to a tsv file
            df = pd.DataFrame(processed_text_features | numerical_features)
            df.to_csv(cache_path, sep="\t", index=False)

        # Preprocess patient-level features
        bag_categorical_features = {
            k: self.scalers["bag_categorical_matching"][k][v]
            if (isinstance(v, str)) or (not isnan(v))
            else 0.0
            for k, v in bag_categorical_features.items()
        }

        bag_numerical_features = {
            k: (v - self.scalers["bag_numerical_scales"][k]["mean"])
            / self.scalers["bag_numerical_scales"][k]["std"]
            if not isnan(v)
            else 0.0
            for k, v in bag_numerical_features.items()
        }

        return {
            "text_features": text_features,
            "numerical_features": numerical_features,
            "bag_categorical_features": bag_categorical_features,
            "bag_numerical_features": bag_numerical_features,
            "label": self.labels[idx],
        }
