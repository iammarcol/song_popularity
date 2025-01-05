from flask import Flask, request, render_template, jsonify
import os
from inference import extract_features_and_predict  # Import the refactored function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    try:
        # Call the feature extraction and prediction function
        results = extract_features_and_predict(file_path)
        
        # Remove the temporary file
        os.remove(file_path)

        # Extract the predicted popularity score as a standard Python float
        predicted_popularity = float(results['predicted_popularity'].iloc[0])  # Ensure it's not a NumPy type
        
        return jsonify({'popularity_score': predicted_popularity})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create uploads folder if not exists
    app.run(debug=True)
