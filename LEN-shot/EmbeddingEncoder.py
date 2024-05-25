from transformers import AutoModel, AutoTokenizer
import torch

class EmbeddingEncoder:
    def __init__(self, model_name, tokenizer_name=None, device=None):
        """
        Initializes the EmbeddingEncoder with a specified model and tokenizer and sets the device for computation.
        
        Args:
            model_name (str): The model identifier from Hugging Face's model hub.
            tokenizer_name (str): The tokenizer identifier. If None, uses model_name.
            device (str): The device to run the model on ('cuda' or 'cpu'). If None, uses cuda if available.
        """
        if tokenizer_name is None:
            tokenizer_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()  # Set the model to evaluation mode

    def encode(self, text_data, output_path=None, output_flag=False, format=False):
        """
        Encodes the provided text data into embeddings using the initialized model.
        
        Args:
            text_data (str or list): Text data to encode. Can be a single string or a list of strings.
            output_path (str): Optional path to save the output.
            output_flag (bool): If True, returns the embeddings directly.
            format (bool): If True, formats the output as JSON.
        
        Returns:
            The embeddings as a list or formatted JSON, depending on the flags.
        """
        # Prepare the text data for the model
        inputs = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Taking the CLS token representation

        if format:
            # Format the embeddings along with text_data into JSON
            formatted_data = [{'text': text, 'embedding': emb.tolist()} for text, emb in zip(text_data, embeddings)]
            if output_path:
                import json
                with open(output_path, 'w') as f:
                    json.dump(formatted_data, f)
            if output_flag:
                return formatted_data
        else:
            if output_path:
                # Save embeddings to a file if required
                import numpy as np
                np.save(output_path, embeddings)
            if output_flag:
                return embeddings

    def format(self, embeddings, text_data, output_path):
        """
        Formats the given embeddings and their corresponding text data into a JSON structure.

        Args:
            embeddings (list): List of embeddings.
            text_data (list): List of text data corresponding to each embedding.
            output_path (str): Path to save the formatted JSON output.
        """
        import json
        formatted_data = [{'text': text, 'embedding': emb.tolist()} for text, emb in zip(text_data, embeddings)]
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f)

# Example usage
if __name__ == "__main__":
    encoder = EmbeddingEncoder('bert-base-uncased')
    text_samples = ["Hello world!", "How are you today?"]
    embeddings = encoder.encode(text_samples, output_flag=True)
    print("Embeddings:", embeddings)

