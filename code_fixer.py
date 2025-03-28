from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_lightweight_fixer_model():
    # Use a small, quantized code generation model
    model_name = "codellama/CodeLlama-7b-Python"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Quantization to reduce model size
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return model, tokenizer

def suggest_fixes(code, bug):
    model, tokenizer = load_lightweight_fixer_model()
    
    # Generate fix suggestion
    prompt = f"Fix the following bug in {code} at line {bug['line']}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=200)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)