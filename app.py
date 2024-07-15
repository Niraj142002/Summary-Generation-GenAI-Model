from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load the tokenizer and model from a local path
model_name_or_path = r"/Users/niraj/Downloads/Intel/saved-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
