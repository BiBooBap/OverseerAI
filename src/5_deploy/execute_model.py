import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from num2words import num2words

# Download delle risorse NLTK richieste
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def multiply_by_0_4(X):
    return X * 0.4

def multiply_by_0_2(X):
    return X * 0.2

# Funzione per normalizzare il testo
def normalize_text(text):
    if not isinstance(text, str) or not text.strip():
        return "Invalid Text"

    contractions = {
        "I'm": "I am", "you're": "you are", "he's": "he is", "she's": "she is", 
        "it's": "it is", "we're": "we are", "they're": "they are", "can't": "cannot", 
        "won't": "will not", "don't": "do not", "didn't": "did not", "isn't": "is not"
    }
    for contraction, full_form in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', full_form, text)

    tokens = word_tokenize(text)
    tokens = [num2words(word) if word.isdigit() else word for word in tokens]
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]

    return ' '.join(tokens)

def extract_link(text):
    link_pattern = r'(http[s]?://\S+)'
    match = re.search(link_pattern, text)
    link = match.group(1) if match else 'no link'
    text_part = re.sub(link_pattern, '', text).strip()
    return text_part, link

# Caricamento del modello salvato
def load_model(filepath):
    return joblib.load(filepath)

# Funzione per prevedere la classe di un titolo
def predict_title(model, title, description):
    normalized_title = normalize_text(title)
    desc_text_part, desc_link = extract_link(description)
    normalized_description = normalize_text(desc_text_part)
    normalized_link_desc = normalize_text(desc_link)

    input_data = pd.DataFrame([{
        'title': normalized_title,
        'description': normalized_description,
        'link_desc': normalized_link_desc
    }])
    prediction = model.predict(input_data)
    return prediction[0]

# Esecuzione
if __name__ == "__main__":
    model_path = "../3_data_modeling/random_forest_model.pkl"
    model = load_model(model_path)

    print("Inserisci il titolo:")
    title = input()
    print("Inserisci la descrizione:")
    description = input()

    result = predict_title(model, title, description)
    print(f"Risultato della classificazione: {result.upper()}")
