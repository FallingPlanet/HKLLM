from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

app = Flask(__name__)

# Initially, there's no model or tokenizer loaded
current_model = None
current_tokenizer = None

initialized = False

def load_configs(path="deploy/models.json"):
    """Loads the model configurations from a specified JSON file."""
    with open(path, 'r') as file:
        models_config = json.load(file)
    return models_config

# Load model configurations from JSON file
model_configs = load_configs("deploy/models.json")

def load_model_config(model_identifier, models_config):
    global current_model, current_tokenizer  # Use global keyword to modify global variables
    model_config = next((m for m in models_config if m["model_identifier"] == model_identifier), None)
    if not model_config:
        raise ValueError(f"Model with identifier {model_identifier} not found.")
    current_tokenizer = AutoTokenizer.from_pretrained(model_config["model_identifier"])
    current_model = AutoModelForCausalLM.from_pretrained(model_config["model_identifier"])
    # Assume adapter_path or other configurations are handled here
    #This can be moved directly from the geenrator in my directory

def initialize_if_needed():
    global initialized, model_configs  # Ensure this uses the loaded configurations and the initialized flag
    if not initialized:
        load_model_config("mistralai/Mistral-7B-Instruct-v0.2", model_configs) 
        initialized = True  # Mark as initialized after the configuration is loaded

@app.before_request
def before_request_func():
    initialize_if_needed()
@app.route('/')
def home():
    # Pass the list of model configs to the template to populate the dropdown
    return render_template('index.html', model_configs=model_configs)

@app.route('/generate', methods=['POST'])
def generate_text():
    global current_model, current_tokenizer
    model_identifier = request.form['model_name']
    try:
        load_model_config(model_identifier, model_configs)
    except ValueError as e:
        return str(e), 404
    user_input = request.form['input_text']
    inputs = current_tokenizer.encode(user_input, return_tensors='pt')
    outputs = current_model.generate(inputs, max_length=200, num_return_sequences=1)
    generated_text = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return render_template('result.html', input_text=user_input, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
