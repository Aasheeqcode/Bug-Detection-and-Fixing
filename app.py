import os
import re
import ast
import torch
import hashlib
import json
import shutil
from typing import Dict, List, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)

# Tkinter and Flask imports
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import webbrowser

class ModelManager:
    """
    Manages model download, caching, and persistence
    """
    def __init__(self, 
                 base_dir='~/.bug_detector_models',
                 model_name='microsoft/CodeGPT-small-py'):
        # Expand and create base directory
        self.base_dir = os.path.expanduser(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Model configuration
        self.model_name = model_name
        self.model_hash = self._generate_model_hash()
        self.model_path = os.path.join(self.base_dir, self.model_hash)
        
        # Model metadata tracking
        self.metadata_file = os.path.join(self.base_dir, 'model_metadata.json')
        
    def _generate_model_hash(self):
        """
        Generate a unique hash for the model
        """
        return hashlib.md5(self.model_name.encode()).hexdigest()
    
    def download_model(self, force_redownload=False):
        """
        Download and cache the model
        """
        # Check if model already exists
        if os.path.exists(self.model_path) and not force_redownload:
            print(f"Model already cached at {self.model_path}")
            return self.model_path
        
        try:
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=self.model_path
            )
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=self.model_path
            )
            
            # Update metadata
            self._update_metadata()
            
            print(f"Model downloaded and cached at {self.model_path}")
            return self.model_path
        
        except Exception as e:
            print(f"Model download failed: {e}")
            raise
    
    def _update_metadata(self):
        """
        Update model download metadata
        """
        try:
            # Read existing metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Update metadata
            metadata[self.model_hash] = {
                'model_name': self.model_name,
                'download_time': str(torch.cuda.current_device()),
                'path': self.model_path
            }
            
            # Write updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        except Exception as e:
            print(f"Metadata update failed: {e}")
    
    def list_cached_models(self):
        """
        List all cached models
        """
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cached_models = []
            for model_hash, details in metadata.items():
                cached_models.append({
                    'name': details['model_name'],
                    'hash': model_hash,
                    'path': details['path'],
                    'download_time': details['download_time']
                })
            
            return cached_models
        
        except FileNotFoundError:
            return []
    
    def clean_cache(self, keep_recent=3):
        """
        Clean model cache, keeping most recent models
        """
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Sort models by download time
            sorted_models = sorted(
                metadata.items(), 
                key=lambda x: x[1]['download_time'], 
                reverse=True
            )
            
            # Remove older models
            for model_hash, details in sorted_models[keep_recent:]:
                model_path = details['path']
                shutil.rmtree(model_path, ignore_errors=True)
                del metadata[model_hash]
            
            # Update metadata file
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return f"Cleaned cache, keeping {keep_recent} most recent models"
        
        except Exception as e:
            return f"Cache cleaning failed: {e}"

class PersistentBugDetector:
    def __init__(
        self, 
        model_name='microsoft/CodeGPT-small-py', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # Model management
        self.model_manager = ModelManager(model_name=model_name)
        
        # Download model if not exists
        model_path = self.model_manager.download_model()
        
        # Device configuration
        self.device = torch.device(device)
        
        # Load model from local cache
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True
        ).to(self.device)
        
        # Generation configuration
        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.max_new_tokens = 150
    
    def detect_and_fix_bugs(self, code: str, language: str):
        """
        Detect and generate fixes for bugs
        """
        # Implement your bug detection logic here
        prompt = f"""
        Analyze the following {language} code for potential bugs:
        
        {code}
        
        Provide detailed bug detection and fix suggestions:
        """
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Generate suggestions
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return suggestions
        suggestions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestions

class BugDetectorGUI:
    def __init__(self, bug_detector):
        self.bug_detector = bug_detector
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Bug Detector Tool")
        self.root.geometry("800x600")
        
        # Create UI components
        self.create_widgets()
        
    def create_widgets(self):
        # Language selection
        tk.Label(self.root, text="Select Language:").pack(pady=5)
        self.language_var = tk.StringVar(value="python")
        languages = ["python", "java", "javascript", "c++"]
        self.language_dropdown = tk.OptionMenu(
            self.root, 
            self.language_var, 
            *languages
        )
        self.language_dropdown.pack(pady=5)
        
        # Code input
        tk.Label(self.root, text="Enter/Paste Code:").pack(pady=5)
        self.code_input = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            width=80, 
            height=10
        )
        self.code_input.pack(pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Detect Bugs Button
        detect_button = tk.Button(
            button_frame, 
            text="Detect Bugs", 
            command=self.detect_bugs
        )
        detect_button.pack(side=tk.LEFT, padx=5)
        
        # Load File Button
        load_button = tk.Button(
            button_frame, 
            text="Load Code File", 
            command=self.load_code_file
        )
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Suggestions display
        tk.Label(self.root, text="Bug Suggestions:").pack(pady=5)
        self.suggestions_output = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            width=80, 
            height=10, 
            state=tk.DISABLED
        )
        self.suggestions_output.pack(pady=10)
        
    def detect_bugs(self):
        # Get code and language
        code = self.code_input.get("1.0", tk.END).strip()
        language = self.language_var.get()
        
        # Validate input
        if not code:
            messagebox.showwarning("Warning", "Please enter code first!")
            return
        
        try:
            # Detect bugs
            suggestions = self.bug_detector.detect_and_fix_bugs(code, language)
            
            # Display suggestions
            self.suggestions_output.config(state=tk.NORMAL)
            self.suggestions_output.delete("1.0", tk.END)
            self.suggestions_output.insert(tk.END, suggestions)
            self.suggestions_output.config(state=tk.DISABLED)
        
        except Exception as e:
            messagebox.showerror("Error", f"Bug detection failed: {str(e)}")
    
    def load_code_file(self):
        # Open file dialog to select code file
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Python Files", "*.py"),
                ("Java Files", "*.java"),
                ("JavaScript Files", "*.js"),
                ("C++ Files", "*.cpp"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    code = file.read()
                    self.code_input.delete("1.0", tk.END)
                    self.code_input.insert(tk.END, code)
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def run(self):
        self.root.mainloop()

def create_flask_app(bug_detector):
    """
    Create Flask backend for web API
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    @app.route('/detect_bugs', methods=['POST'])
    def detect_bugs():
        data = request.json
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        try:
            suggestions = bug_detector.detect_and_fix_bugs(code, language)
            return jsonify({
                'suggestions': suggestions,
                'status': 'success'
            })
        
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500
    
    @app.route('/list_models', methods=['GET'])
    def list_models():
        models = bug_detector.model_manager.list_cached_models()
        return jsonify(models)
    
    @app.route('/clean_cache', methods=['POST'])
    def clean_cache():
        result = bug_detector.model_manager.clean_cache()
        return jsonify({'message': result})
    
    return app

def start_flask_server(app, port=5000):
    """
    Start Flask server in a separate thread
    """
    def run_server():
        app.run(host='0.0.0.0', port=port, debug=False)
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open web browser
    webbrowser.open(f'http://localhost:{port}')

def main():
    # Initialize bug detector
    bug_detector = PersistentBugDetector()
    
    # Create Flask app
    flask_app = create_flask_app(bug_detector)
    
    # Start Flask server
    start_flask_server(flask_app)
    
    # Launch Tkinter GUI
    gui = BugDetectorGUI(bug_detector)
    gui.run()

if __name__ == "__main__":
    main()