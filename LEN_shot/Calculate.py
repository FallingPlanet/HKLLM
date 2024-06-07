
import numpy as np
import math

class Calculate:
    def __init__(self, method='cosine'):
        """
        Initializes the Calculate instance with a default method for distance or similarity calculation.
        
        Args:
            method (str): The method to use for calculations ('cosine', 'euclidean', 'manhattan', 'angular', 'jensenshannon').
        """
        self.method = method

    def compute(self, vec1, vec2, method=None):
        """
        Computes the similarity or distance between two vectors based on a specified method. Optionally overrides the default method.
        
        Args:
            vec1 (np.array): First vector.
            vec2 (np.array): Second vector.
            method (str, optional): Method to use for this particular computation; if None, uses the method defined at initialization.
        
        Returns:
            float: The computed similarity or distance.
        """
        # Use the instance method if no method is specified for this call
        if method is None:
            method = self.method

        if method == 'cosine':
            return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        elif method == 'euclidean':
            return np.linalg.norm(vec1 - vec2)
        elif method == 'manhattan':
            return np.sum(np.abs(vec1 - vec2))
        elif method == 'angular':
            unit_vec1 = vec1 / np.linalg.norm(vec1)
            unit_vec2 = vec2 / np.linalg.norm(vec2)
            angle = math.acos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))
            return angle
        elif method == 'jensenshannon':
            # Jensen-Shannon distance calculation
            def normalize(v):
                total = np.sum(v)
                return v / total if total != 0 else v
            vec1_normalized = normalize(vec1)
            vec2_normalized = normalize(vec2)
            m = 0.5 * (vec1_normalized + vec2_normalized)
            kl_div1 = np.sum(vec1_normalized * np.log(vec1_normalized / m + 1e-10))
            kl_div2 = np.sum(vec2_normalized * np.log(vec2_normalized / m + 1e-10))
            js_distance = 0.5 * (kl_div1 + kl_div2)
            return math.sqrt(js_distance)
        else:
            raise ValueError(f"Unknown method: {method}")

