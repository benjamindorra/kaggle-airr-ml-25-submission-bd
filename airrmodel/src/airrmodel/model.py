from torch import nn, uniform
from transformers import AutoTokenizer, EsmModel


class EsmABMIL:
    """
    This class implements the EsmABMIL model.
    Input texts giving an antibody genetic structure, modifications, and other relevant informations
    are encoded using an ESM transformer model.
    Dropout is used on the embeddings to allow for missing features
    and exploit the incomplete information in the dataset.
    Then the concatenated features are aggregated through additive attention.
    Finally a classification layer outputs a result for the patient.
    """

    def __init__(
        self,
        attention_dim,
        feature_extraction_model,
        head_dim,
        features_dropout,
        num_classes,
        num_features,
    ):
        self.attention_dim = attention_dim
        self.head_dim = head_dim
        self.features_dropout = features_dropout
        self.num_classes = num_classes
        self.num_features = num_features
        self.tokenizer = AutoTokenizer.from_pretrained(feature_extraction_model)
        self.feature_extractor = EsmModel.from_pretrained(feature_extraction_model)
        self.embed_dim = self.feature_extractor.config.hidden_size
        self.attention_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.attention_dim),
            nn.Tanh(),
            nn.Linear(self.attention_dim, 1),
        )
        self.classification_head = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, inputs):
        # Pretrained encoder
        with torch.no_grad():
            # Encode input using ESM transformer model
            # Expect list[list[str]] input, outputs list[torch.Tensor] with dim [num_features, hidden_size]
            outputs = self.tokenizer(inputs)
            embeddings = outputs.last_hidden_state

            # Apply dropout to embeddings
            # Important: applied to whole embeddings tensors, sets a feature to 0 with probability features_dropout
            keep_features = [
                (
                    uniform(
                        self.num_features,
                    )
                    > self.features_dropout
                ).unsqueeze(1)
                for _ in range(len(embeddings))
            ]
            embeddings = embeddings * keep_features

        # Aggregate features using additive attention
        attention_weights = nn.functional.softmax(
            self.attention_head(embeddings), dim=-1
        )
        aggregated_features = attention_weights * embeddings
        aggregated_features = aggregated_features.sum()

        # Output classification result
        logits = self.classification_head(aggregated_features)
        return logits
