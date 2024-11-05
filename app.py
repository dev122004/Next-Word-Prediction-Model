from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained LSTM model
model = load_model('lstm_model_2.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 20  # Set max sequence length (used for padding)

def predict_next_words(text, num_words=3):

    for _ in range(num_words):
        # Tokenize and pad the input text
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
        
        # Predict the next word index
        predicted_index = np.argmax(model.predict(padded_token_text), axis=-1)[0]
        
        # Map the index to the word
        next_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break
        
        # If we can't find a word, stop predicting
        if not next_word:
            break
        
        # Append the predicted word to the input text
        text += ' ' + next_word
    
    return text  # Return the full sentence with predicted words

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text and desired number of words from the form
    text = request.form['text']
    num_words = int(request.form.get('num_words', 3))  # Default to 3 if not specified
    
    # Predict the next words
    predicted_text = predict_next_words(text, num_words=num_words)
    
    return render_template('index.html', input_text=text, prediction=predicted_text)

if __name__ == "__main__":
    app.run(debug=True)
