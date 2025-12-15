
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class RepertoireDataset(Dataset):
    """
    Dataset class for AIRR repertoires.
    Reads metadata and loads corresponding sequences from TSV files.
    """
    def __init__(self, metadata_path, data_dir, model_name="facebook/esm2_t12_35M_UR50D", max_length=128, sample_size=None):
        """
        Args:
            metadata_path (str): Path to the metadata.csv file.
            data_dir (str): Directory containing the .tsv files.
            model_name (str): Name of the ESM2 model for tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            sample_size (int, optional): if set, randomly sample this many sequences from each larger repertoire.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.sample_size = sample_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        # Handle cases where label might not exist (test set)
        label = row['label_positive'] if 'label_positive' in row else -1
        
        # Load TSV
        file_path = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Return empty or dummy if file read fails - though validation should catch this
            return None

        # Extract sequences (junction_aa)
        # Drop NaNs or invalid sequences if necessary
        sequences = df['junction_aa'].dropna().astype(str).tolist()
        
        # Sampling if repertoire is too large
        if self.sample_size and len(sequences) > self.sample_size:
            sequences = pd.Series(sequences).sample(n=self.sample_size, random_state=42).tolist()
            
        # Tokenize
        # We process a list of strings. ESM tokenizer handles this relative to standard BERT.
        # But commonly we might want to do it in collate to batch properly, 
        # or do it here. Doing it here allows caching logic later if needed.
        # For variable bag sizes, we usually return a list of tensors or a pre-padded tensor.
        # Here we will return the raw encoded inputs for the bag.
        
        encoding = self.tokenizer(
            sequences, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # encoding['input_ids'] is (num_sequences, max_length)
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.float),
            'id': row['repertoire_id'] if 'repertoire_id' in row else filename
        }

def collate_fn(batch):
    """
    Collate function for batches of repertoires (bags).
    Since each item is a bag of sequences (tensor of shape [N_seq, L]), 
    and N_seq varies, we cannot simply stack them into [Batch, N_seq, L].
    
    We can either:
    1. Return a list of tensors (easiest for custom training loop).
    2. Concat all sequences into one big batch [Total_Seq, L] and keep track of which bag they belong to.
    
    Structure 1 is chosen for clarity in MIL.
    """
    # Filter out Nones
    batch = [item for item in batch if item is not None]
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    ids = [item['id'] for item in batch]
    
    return {
        'input_ids': input_ids,       # List of tensors
        'attention_mask': attention_mask, # List of tensors
        'labels': labels,             # Tensor [Batch]
        'ids': ids                    # List of strings
    }
