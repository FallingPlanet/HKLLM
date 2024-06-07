from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from .Formatter import llama_3_formatting_func  # Ensure correct import path

class EmbeddingEncoder:
    def __init__(self, model, tokenizer_name=None, device=None, adapter_name=None, use_llama3_format=False):
        """Initializes the EmbeddingEncoder with specific configurations for the model, tokenizer, and formatting function.

        Args:
        model (str or torch.nn.Module): The pre-trained model or the path to the pre-trained model directory.
        tokenizer_name (str, optional): The tokenizer name or path used for tokenizing input texts.
        device (str, optional): The device type ('cuda' or 'cpu') on which the model should be loaded.
        adapter_name (str, optional): The path or identifier for a pre-trained adapter to be loaded into the model.
        use_llama3_format (bool, optional): Flag to determine whether to preprocess text data using the Llama-3 formatting function before encoding.

        Sets up the model and tokenizer and ensures that the model is in evaluation mode. Adapters are loaded if specified.
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        token = os.getenv('HF_TOKEN')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=token, device_map="auto", output_hidden_states=True)
            if adapter_name:
                adapter_source = 'hf' if os.path.exists(adapter_name) else 'local'
                self.model.load_adapter(adapter_name, source=adapter_source)
        else:
            self.model = model.to(self.device)

        # Initialize the tokenizer
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.model.eval()

        # Setting up the formatting function
        
        self.formatting_func = llama_3_formatting_func if use_llama3_format else None

    def encode(self, text_data, output_path=None, return_format='numpy', batch_size=16):
        """
        Encodes text data into embeddings using the model's last hidden states.

        Args:
            text_data (list or dict): The text data to encode. Can be a list of strings or a dictionary with text data.
            output_path (str, optional): Path to save the output embeddings if specified.
            return_format (str, optional): The format to return the embeddings ('numpy', 'list', or 'json').
            batch_size (int, optional): The batch size to use for processing texts through the model.

        Returns:
            numpy.ndarray, list, or dict: The embeddings in the specified format. If 'json' is chosen and output_path is specified, data is also saved as a JSON file.

        Processes the text through the formatting function if set, tokenizes the text, and feeds it into the model to get embeddings, which are then returned in the specified format.

        """
        if self.formatting_func:
            print("Type of text_data:", type(text_data))
            print("Content of text_data:", text_data[:5])  # Print first few entries to avoid too much output
            # Example debug snippet
            if isinstance(text_data, list) and all(isinstance(i, dict) for i in text_data):
                print("Data is correctly formatted.")
            else:
                print("Data is not correctly formatted. Check the structure.")

            text_data = self.formatting_func(text_data, convert_from_instruction=False,convert_from_prompt_only=True, bos_token="<|startoftext|>", eos_token="<|endoftext|>")
        
        inputs = self.tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        dataset = TensorDataset(inputs['input_ids'], inputs.get('attention_mask', torch.Tensor()))
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_embeddings = []
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device) if attention_mask.nelement() != 0 else None
            # Ensure that the attention_mask is not None and has been correctly processed
            if attention_mask is not None and attention_mask.nelement() != 0:
                model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                model_inputs = {'input_ids': input_ids}


            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1][:, 0, :]
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        if return_format == 'json':
            formatted_data = [{'text': text, 'embedding': emb.tolist()} for text, emb in zip(text_data, all_embeddings)]
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(formatted_data, f)
            return formatted_data
        elif return_format == 'list':
            return all_embeddings.tolist()
        else:
            if output_path:
                np.save(output_path, all_embeddings)
            return all_embeddings

    @staticmethod
    def adapt_rlhf_data(rlhf_data):
        """
        Adapts data from RLHF format to a more general format with 'prompt' and 'completion' fields.

        Args:
            rlhf_data (dict or list of dicts): Data in RLHF format which typically contains 'prompt', 'chosen', and 'rejected' fields.

        Returns:
            dict or list of dicts: Reformatted data with 'prompt' and 'completion' fields where 'completion' is derived from the 'chosen' field in the original data.

        This function is useful for converting RLHF-specific formatted data into a format suitable for general use, particularly in training or inference setups.

                """
        if isinstance(rlhf_data, dict):
            return {"prompt": rlhf_data['prompt'], "completion": rlhf_data['chosen']}
        elif isinstance(rlhf_data, list):
            return [{"prompt": item['prompt'], "completion": item['chosen']} for item in rlhf_data]
        else:
            raise TypeError("Expected rlhf_data to be a dict or list of dicts")


