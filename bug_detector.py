from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_lightweight_model():
    # Use a small, quantized model
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Quantization to reduce model size
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model, tokenizer

def detect_bugs(code, language):
    model, tokenizer = load_lightweight_model()
    
    # Tokenize and process code
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict bugs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process and return bug information
    return parse_bug_predictions(outputs)

def parse_bug_predictions(model_outputs):
    # Implement logic to convert model outputs to bug details
    # Return list of bug dictionaries
    pass