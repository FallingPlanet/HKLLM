import numpy as np
from Calculate import Calculate
import json

class ChatTemplateCreator:
    def __init__(self, tokenizer, calculate, embedding_path, num_examples=3):
        """
        Initializes the ChatTemplateCreator.
        
        Args:
            tokenizer (AutoTokenizer): Tokenizer for processing texts.
            calculate (Calculate): Instance for computing similarities.
            embedding_path (str): Path to the .npy file containing embeddings.
            num_examples (int): Number of n-shot examples to include in the prompt.
        """
        self.tokenizer = tokenizer
        self.calculate = calculate
        self.embeddings = np.load(embedding_path)
        self.num_examples = num_examples

    def create_template(self, data, system_prompt, user_prompt, follow_up_prompt):
        """
        Creates a formatted chat template using n-shot learning examples based on calculated similarities.
        
        Args:
            data (dict): Dictionary containing keys 'text' and 'response' with lists as values.
            system_prompt (str): System prompt introducing the task.
            user_prompt (str): User prompt following the examples.
            follow_up_prompt (str): Follow-up prompt to complete the interaction.

        Returns:
            str: A formatted chat template.
        """
        # Ensure data alignment with embeddings
        if len(data['text']) != len(self.embeddings):
            raise ValueError("Mismatch between the number of embeddings and the number of data points in the data.")

        distances = self._calculate_distances()
        selected_indices = self._select_indices(distances)

        examples_text = self._format_examples(data, selected_indices)
        full_text = f"{system_prompt}\n{examples_text}\n{user_prompt}\n{follow_up_prompt}"

        return full_text

    def _calculate_distances(self):
        distances = []
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                distance = self.calculate.compute(self.embeddings[i], self.embeddings[j])
                distances.append((distance, i, j))
        distances.sort()
        return distances

    def _select_indices(self, distances):
        selected_indices = set()
        for _, i, j in distances[:self.num_examples]:
            selected_indices.update([i, j])
            if len(selected_indices) >= self.num_examples:
                break
        return selected_indices

    def _format_examples(self, data, selected_indices):
        formatted_examples = []
        for idx in sorted(list(selected_indices)[:self.num_examples]):
            text = data['text'][idx]
            response = data['response'][idx]
            formatted_examples.append(f"###Example {idx+1}###\nText:\n{text}\n\nResponse:\n{response}")
        return "\n".join(formatted_examples)
# Example usage
calculate = Calculate(method='cosine')
tokenizer = None  # Assuming tokenizer is set up elsewhere
template_format = "[INST] You have the following examples as reference for classification:\n"

creator = ChatTemplateCreator(calculate, tokenizer, template_format)
data = {
    'x': ["Text sample 1", "Text sample 2", "Text sample 3", "Text sample 4"],
    'y': [np.array([0.1, 0.2]), np.array([0.2, 0.1]), np.array([0.1, 0.3]), np.array([0.3, 0.1])]
}
system_prompt = "System Introduction:"
user_prompt = "User question?"
follow_up_prompt = "Please provide your answer."
template = creator.create_template(data, system_prompt, user_prompt, follow_up_prompt)
print(template)
