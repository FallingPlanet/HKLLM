import numpy as np
from .Calculate import Calculate
from .EmbeddingEncoder import EmbeddingEncoder
import json

class ChatTemplateCreator:
    def __init__(self, embedding_encoder, calculate, tokenizer, embedding_path, num_examples=3, example_role='assistant', llama_3=True, **kwargs):
        """
        Initializes the ChatTemplateCreator with an EmbeddingEncoder, a Calculate instance, and a tokenizer for embedding, distance calculations, and text formatting.
        
        Args:
            embedding_encoder (EmbeddingEncoder): An instance of EmbeddingEncoder for generating embeddings.
            calculate (Calculate): An instance of Calculate for computing distances.
            tokenizer: Tokenizer for processing text inputs.
            embedding_path (str): Path to the .npy file containing precomputed embeddings.
            num_examples (int): Number of n-shot examples to include in the prompt.
            example_role (str): Role for the n-shot examples ('user', 'assistant', or 'alternate').
            llama_3 (bool): Whether to use Llama-3 chat template formatting.
            **kwargs: Additional keyword arguments that can include 'distance_type'.
        """
        self.embedding_encoder = embedding_encoder
        self.calculate = calculate
        self.tokenizer = tokenizer
        self.embeddings = np.load(embedding_path)
        self.num_examples = num_examples
        self.example_role = example_role
        self.llama_3 = llama_3
        self.distance_type = kwargs.get('distance_type', 'cosine')  # Default to 'cosine' if not specified

    def create_template(self, text, system_prompt, user_prompt, follow_up_prompt):
        """
        Creates a formatted chat template using a target text to find the most similar examples based on embedding distances.
        """
        target_embedding = self.embedding_encoder.encode([text])[0]
        distances = [(self.calculate.compute(target_embedding, emb, self.distance_type), idx) for idx, emb in enumerate(self.embeddings)]
        selected_indices = [idx for _, idx in sorted(distances)[:self.num_examples]]
        examples_text = self.format_examples(selected_indices)

        return self.assemble_chat_template(examples_text, text, system_prompt, user_prompt, follow_up_prompt)

    def format_examples(self, selected_indices):
        """
        Formats selected examples based on indices.
        """
        formatted_examples = []
        for idx in selected_indices:
            prompt = self.embeddings[idx]['prompt']
            completion = self.embeddings[idx]['completion']
            if self.example_role == 'alternate':
                formatted_examples.append({"role": "user", "content": prompt})
                formatted_examples.append({"role": "assistant", "content": completion})
            else:
                role = "assistant" if self.example_role == 'assistant' else "user"
                formatted_examples.append({"role": role, "content": f"{prompt}\n{completion}"})
        return formatted_examples

    def assemble_chat_template(self, examples_text, user_input, system_prompt, user_prompt, follow_up_prompt):
        """
        Assembles the complete chat template including system prompts and user interactions.
        """
        if self.llama_3:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": user_input}
            ] + examples_text
            messages.append({"role": "user", "content": follow_up_prompt})
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            examples_section = "\n".join([ex['content'] for ex in examples_text])
            full_text = f"{system_prompt}\n{user_prompt}\n{user_input}\n{examples_section}\n{follow_up_prompt}"
        return full_text
