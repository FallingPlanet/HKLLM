# Documentation for `EmbeddingEncoder`

## Overview
`EmbeddingEncoder` is a key component of the LEN-shot project, designed to handle the generation and formatting of embeddings from text data. This module supports JSON input files and dictionaries, ensuring flexibility and compatibility with typical dataset formats.

## Functions

### `encode()`
The `encode()` function processes text data to generate embeddings. It can accept data in JSON format or as a dictionary and supports optional parameters to control output behavior and automatic formatting.

**Parameters:**
- `input_data`: Path to the JSON file or dictionary containing the text data.
- `output_path`: Optional; if provided, the embeddings or formatted data are saved to this path.
- `output_flag`: Optional; boolean, default `False`. If `True`, returns the embeddings directly.
- `format`: Optional; boolean, default `False`. If `True`, performs formatting on the embeddings and returns or saves a JSON dictionary of embeddings along with their corresponding text data.

**Returns:**
- Depending on `output_flag` and `format` settings, returns a list of embeddings or a formatted JSON dictionary.

**Usage Example:**
```python
from len_shot import EmbeddingEncoder

encoder = EmbeddingEncoder()

For encoding only
embeddings = encoder.encode('path/to/input.json', output_flag=True)
print("Generated Embeddings:", embeddings)

For encoding and formatting directly
formatted_data = encoder.encode('path/to/input.json', format=True, output_flag=True)
print("Formatted Data:", formatted_data)
```

### `format()`
The `format()` function organizes embeddings, along with their corresponding text data, into a structured JSON format suitable for further processing or model training. This function is useful if data needs to be formatted after encoding has been done separately.

**Parameters:**
- `embeddings`: List of embeddings.
- `text_data`: List of dictionaries representing the original text data corresponding to each embedding.
- `output_path`: Path to save the formatted JSON output.

**Returns:**
- None; outputs a JSON file at the specified `output_path` containing the embeddings and text data.

**Usage Example:**
```python
from len_shot import EmbeddingEncoder

Assuming embeddings and text_data are already defined
encoder = EmbeddingEncoder()
encoder.format(embeddings, text_data, 'path/to/output.json')
```
