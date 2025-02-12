from flask import Flask, render_template, request
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
feature_extraction = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define the main route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_mail = [request.form['message']]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)
    
    result = "Ham Mail" if prediction[0] == 1 else "Spam Mail"
    return render_template('index.html', prediction_text=f'The message is: {result}')

if __name__ == "__main__":
    app.run(debug=True)
