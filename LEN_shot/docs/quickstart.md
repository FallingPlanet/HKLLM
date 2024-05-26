## Quickstart Guide for "LEN-shot"

### Prerequisites
- Python (version < 3.12) is required due to package compatibility issues with newer versions.
- Git installed on your machine.

### Installation

**Step 1: Clone the HKLLM Repository**

To get started with LEN-shot, you will need to clone the entire HKLLM repository, which includes LEN-shot. Open your terminal and run:
```bash
git clone https://github.com/FallingPlanet/HKLLM.git
cd HKLLM/LEN-shot
```

**Step 2: Install Dependencies**

Once you are in the LEN-shot directory, install the required Python dependencies by running:

```bash
pip install -r requirements.txt
```

**Step 3: Installation via pip (for compatible Python versions)**

If your project setup allows, and for Python versions less than 3.12, LEN-shot might be available for installation via pip directly (this depends on how you package and distribute it):
```bash
pip install len-shot
```
Otherwise, After cloning HKLLM to the working directory, 
### Using LEN-shot Locally

If LEN-shot is not installed via pip, you can still use it directly from the cloned directory by appending the path to your Python system paths. This allows you to import and use LEN-shot as if it were installed. In your Python script or interactive session, you can set it up like this:

```python
import sys
sys.path.append('/path/to/HKLLM/LEN-shot')  # Adjust this path as necessary
```

### Example Usage
```python
# Import the embedding encoder from LEN-shot

from len_shot import EmbeddingEncoder

# Example usage of the EmbeddingEncoder
encoder = EmbeddingEncoder()
text = "Example text to encode"
embedding = encoder.encode(text)
print("Generated Embedding:", embedding)
```
