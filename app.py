from flask import Flask, request, jsonify
from ml_models.bug_detector import detect_bugs
from ml_models.code_fixer import suggest_fixes

app = Flask(__name__)

@app.route('/detect_bugs', methods=['POST'])
def process_code():
    data = request.json
    code = data['code']
    language = data['language']
    
    # Detect bugs
    bugs = detect_bugs(code, language)
    
    # Generate fixes
    for bug in bugs:
        bug['fix'] = suggest_fixes(code, bug)
    
    return jsonify({
        'bug_count': len(bugs),
        'bugs': bugs
    })

if __name__ == '__main__':
    app.run(port=5000)