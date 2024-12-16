from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Define the path to the model
MODEL_PATH = os.path.join('models', 'xgboost_heart_model.pkl')

# Load the trained XGBoost model
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """
    Render the homepage with the prediction form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the prediction request from the form.
    """
    try:
        # Parse input features from the form
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0, 1]
        
        # Prepare result for display
        result = {
            'prediction': 'Disease Present' if prediction == 1 else 'No Disease',
            'probability': f"{probability * 100:.2f}%"
        }
        return render_template('index.html', result=result)
    except Exception as e:
        # Handle any errors during prediction
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    """
    Run the Flask app.
    """
    app.run(debug=True)
