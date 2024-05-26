from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import json

class EmbeddingEncoder:
    def __init__(self, model_name, tokenizer_name=None, device=None,token=None):
        """
        Initializes the EmbeddingEncoder with a specified model and tokenizer and sets the device for computation.
        
        Args:
            model_name (str): The model identifier from Hugging Face's model hub.
            tokenizer_name (str): Optional; the tokenizer identifier. If None, uses model_name.
            device (str): Optional; the device to run the model on ('cuda' or 'cpu'). If None, uses cuda if available.
        """
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name,token=token).to(self.device)
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
        # Handling different types of text_data input
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

# Example usage
if __name__ == "__main__":
    encoder = EmbeddingEncoder('mistralai/Mistral-7B-Instruct-v0.3',token='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')
    text_samples = ["Hello world!", "How are you today?"]
    embeddings = encoder.encode(text_samples, return_format='list')
    print("Embeddings:", embeddings)


