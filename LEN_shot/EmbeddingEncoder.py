from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import json
import os

class EmbeddingEncoder:
    def __init__(self, model, tokenizer_name=None, device=None, adapter_name=None):
        # Retrieve token from environment variable
        token = os.getenv('HF_TOKEN')
        
        # Set up tokenizer name based on model if not provided
        if tokenizer_name is None:
            tokenizer_name = model if isinstance(model, str) else None
        
        # Setup device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model from Hugging Face hub or a local model
        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model, token=token).to(self.device)
            # Load adapter if specified
            if adapter_name:
                adapter_source = 'hf' if os.path.exists(adapter_name) else 'local'
                self.model.load_adapter(adapter_name, source=adapter_source)
        else:
            # If a model instance is provided, just use it
            self.model = model.to(self.device)

        # Load tokenizer
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set model to evaluation mode
        self.model.eval()

    def encode(self, text_data, output_path=None, return_format='numpy', text_column=None):
        """
        Encodes the provided text data into embeddings using the initialized model.
        
        Args:
            text_data (str, list, or dict): Text data to encode. Can be a single string, list of strings, or a JSON-like dictionary.
            output_path (str): Optional path to save the output (JSON or NumPy file).
            return_format (str): The format of the return value ('numpy', 'json', or 'list').
            text_column (str): Optional; key to use if text_data is a dictionary (for JSON-like structures).
        
        Returns:
            Embeddings as specified by the return format.
        """
        if isinstance(text_data, dict):
            if text_column is None:
                raise ValueError("text_column must be specified if text_data is a dictionary")
            text_data = [item[text_column] for item in text_data]
        elif isinstance(text_data, str):
            text_data = [text_data]  # Convert a single string to a list

        # Prepare the text data for the model
        inputs = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract the CLS token representation

        # Handle different return formats
        if return_format == 'json':
            formatted_data = [{'text': text, 'embedding': emb.tolist()} for text, emb in zip(text_data, embeddings)]
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(formatted_data, f)
            return formatted_data
        elif return_format == 'list':
            return embeddings.tolist()
        else:
            if output_path:
                np.save(output_path, embeddings)
            return embeddings

    @staticmethod
    def adapt_rlhf_data(rlhf_data):
        """
        Adapts RLHF data to a format suitable for other components, converting 'chosen' to 'completion'.
        It operates non-destructively, returning a new dictionary with the desired structure.
        
        Args:
            rlhf_data (dict or list of dicts): RLHF data in the format with 'prompt', 'chosen', and 'rejected'.

        Returns:
            dict or list of dicts: New data formatted with 'prompt' and 'completion' keys, original data remains unchanged.
        """
        if isinstance(rlhf_data, dict):
            # Handle single dictionary case, create a new dictionary
            return {
                "prompt": rlhf_data['prompt'],
                "completion": rlhf_data['chosen']
            }
        elif isinstance(rlhf_data, list):
            # Handle list of dictionaries case, create new dictionaries for each item
            return [{
                "prompt": item['prompt'],
                "completion": item['chosen']
            } for item in rlhf_data]
        else:
            raise TypeError("Expected rlhf_data to be a dict or list of dicts")

