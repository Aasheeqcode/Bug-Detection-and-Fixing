import tkinter as tk
from tkinter import filedialog, messagebox
import requests

class BugDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Code Bug Detector")
        
        # Language Selection
        self.language_var = tk.StringVar()
        self.languages = ['Python', 'Java', 'C++', 'JavaScript']
        
        self.create_widgets()
    
    def create_widgets(self):
        # File Selection
        tk.Button(self.root, text="Select Code File", 
                  command=self.select_file).pack(pady=10)
        
        # Language Dropdown
        tk.Label(self.root, text="Select Programming Language:").pack()
        language_dropdown = tk.OptionMenu(self.root, self.language_var, *self.languages)
        language_dropdown.pack(pady=10)
        
        # Verify Button
        tk.Button(self.root, text="Detect Bugs", 
                  command=self.detect_bugs).pack(pady=10)
        
        # Results Display
        self.results_text = tk.Text(self.root, height=15, width=70)
        self.results_text.pack(pady=10)
    
    def select_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("Code Files", f"*.{self.language_var.get().lower()}")
        ])
    
    def detect_bugs(self):
        if not hasattr(self, 'filename'):
            messagebox.showerror("Error", "Please select a file first")
            return
        
        with open(self.filename, 'r') as file:
            code_content = file.read()
        
        try:
            response = requests.post('http://localhost:5000/detect_bugs', 
                                     json={
                                         'code': code_content, 
                                         'language': self.language_var.get()
                                     })
            results = response.json()
            
            self.display_results(results)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def display_results(self, results):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Bugs Detected: {results['bug_count']}\n\n")
        
        for bug in results['bugs']:
            self.results_text.insert(tk.END, f"Bug Type: {bug['type']}\n")
            self.results_text.insert(tk.END, f"Location: Line {bug['line']}\n")
            self.results_text.insert(tk.END, f"Description: {bug['description']}\n")
            self.results_text.insert(tk.END, f"Suggested Fix:\n{bug['fix']}\n\n")

def main():
    root = tk.Tk()
    app = BugDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()