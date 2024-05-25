import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock, jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

class Calculate:
    def __init__(self, method='cosine'):
        self.method = method

    def compute(self, vec1, vec2):
        """
        Computes the similarity or distance between two vectors based on the specified method.

        Args:
            vec1 (np.array): First vector.
            vec2 (np.array): Second vector.

        Returns:
            float: The computed similarity or distance.
        """
        if self.method == 'cosine':
            return 1 - cosine(vec1, vec2)
        elif self.method == 'euclidean':
            return euclidean(vec1, vec2)
        elif self.method == 'manhattan':
            return cityblock(vec1, vec2)
        elif self.method == 'rbf_kernel':
            return rbf_kernel(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        elif self.method == 'polynomial_kernel':
            return polynomial_kernel(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        elif self.method == 'angular':
            unit_vec1 = vec1 / np.linalg.norm(vec1)
            unit_vec2 = vec2 / np.linalg.norm(vec2)
            return np.arccos(np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0))
        elif self.method == 'emd':
            return wasserstein_distance(vec1, vec2)
        elif self.method == 'jensenshannon':
            # Normalize vectors for Jensen-Shannon
            def normalize(v):
                return v / np.sum(v)
            vec1_normalized = normalize(vec1)
            vec2_normalized = normalize(vec2)
            return jensenshannon(vec1_normalized, vec2_normalized)
        else:
            raise ValueError(f"Unknown method: {self.method}")

# Example Usage
if __name__ == "__main__":
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])

    calculator = Calculate(method='cosine')
    print("Cosine Similarity:", calculator.compute(vec1, vec2))

    calculator.method = 'euclidean'
    print("Euclidean Distance:", calculator.compute(vec1, vec2))

    calculator.method = 'emd'
    print("Earth Mover's Distance:", calculator.compute(vec1, vec2))
