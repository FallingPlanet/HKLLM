from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import json
import os

class EmbeddingEncoder:
    def __init__(self, model, tokenizer_name=None, device=None, adapter_name=None):
        token = os.getenv('HF_TOKEN')
        """
        Initializes the EmbeddingEncoder with a specified model or model name, an optional tokenizer, and sets the device for computation.
        
        Args:
            model (Union[str, torch.nn.Module]): The model identifier from Hugging Face's model hub or a model instance.
            tokenizer_name (str): Optional; the tokenizer identifier. If None, uses model if it's a string.
            device (str): Optional; the device to run the model on ('cuda' or 'cpu'). If None, uses cuda if available.
            adapter_name (str): Optional; the adapter identifier to load with the model, either a path or from the Hugging Face hub.
        """
        if tokenizer_name is None:
            tokenizer_name = model if isinstance(model, str) else None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(model, str):
            self.model = AutoModel.from_pretrained(model, token=token).to(self.device)
            if adapter_name:
                # Check if the adapter path exists or it is from the hub
                if os.path.exists(adapter_name):
                    # Load adapter from the local path
                    self.model.load_adapter(adapter_name)
                else:
                    # Load adapter from Hugging Face Hub
                    self.model.load_adapter(adapter_name, source='hf')
        else:
            self.model = model.to(self.device)

        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()  # Set the model to evaluation mode

    def encode(self, text_data, output_path=None, return_format='numpy', text_column=None):
        """
        Encodes the provided text data into embeddings using the initialized model.
        
        Args:
            text_data (str, list, or dict): Text data to encode. Can be a single string, list of strings, or a JSON-like dictionary.
            output_path (str): Optional path to save the output (json or numpy file).
            return_format (str): The format of the return value ('numpy', 'json', or 'list').
            text_column (str): Optional; key to use if text_data is a dictionary (for JSON-like structures).
        
        Returns:
            Embeddings as specified by return_format.
        """
        # Handling different types of text data input
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
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Taking the CLS token representation

        # Handling different return formats
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
    def adapt_rlhf_data(rlhf_data):
        """
        Adapts RLHF data to a format suitable for other components, by converting 'chosen' to 'completion'.
        It operates non-destructively, returning a new dictionary with the desired structure.
        
        Args:
            rlhf_data (dict or list of dicts): RLHF data in the format with 'prompt', 'chosen', and 'rejected'.

        Returns:
            dict or list of dicts: New data formatted with 'prompt' and 'completion' keys, original data remains unchanged.
        """
        if isinstance(rlhf_data, dict):
            # Handle single dictionary case, create a new dictionary
            adapted_dict = {
                "prompt": rlhf_data['prompt'],
                "completion": rlhf_data['chosen']
            }
            return adapted_dict
        elif isinstance(rlhf_data, list):
            # Handle list of dictionaries case, create new dictionaries for each item
            adapted_list = []
            for item in rlhf_data:
                adapted_item = {
                    "prompt": item['prompt'],
                    "completion": item['chosen']
                }
                adapted_list.append(adapted_item)
            return adapted_list
        else:
            raise TypeError("Expected rlhf_data to be a dict or list of dicts")

    
