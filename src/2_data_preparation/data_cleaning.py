import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from num2words import num2words

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Inizializza strumenti
stop_words = set(stopwords.words('english'))
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()

# Data Cleaning pipeline
def clean_dataset(df):
    # Valori mancanti.
    # Imputazione dei valori mancanti per le colonne numeriche con la mediana.
    def value_imputation(df, column):
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

        return df

    for col in ['upload_hour', 'likes', 'dislikes', 'comments']:
        df = value_imputation(df, col)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[['likes','dislikes','comments']])
    plt.title("Line plots before removing anomalies")
    plt.show()

    # Rimuovi i valori anomali per le colonne 'likes', 'dislikes' e 'comments' utilizzando la funzione interquartile range (IQR).
    def remove_anomalies(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median_val = df[column].median()
        df.loc[df[column] < lower_bound, column] = median_val
        df.loc[df[column] > upper_bound, column] = median_val
        return df
    
    for col in ['likes', 'dislikes', 'comments']:
        df = remove_anomalies(df, col)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[['likes','dislikes','comments']])
    plt.title("Line plots after removing anomalies")
    plt.show()
    
    # Pulizia dei titoli.
    def clean_title(title):
        if pd.isnull(title) or '###ERROR###' in str(title):
            return "Title Not Available"

        # Rimuovi caratteri speciali indesiderati.
        title = re.sub(r'[^\w\s]', '', str(title))
        # Rimuovi spazi multipli.
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def process_description(description):
    # Cerca un link nella descrizione
        link = re.search(r'http[s]?://\S+', description)
        return link.group() if link else "No link available"
    
    df['title'] = df['title'].apply(clean_title)
    df['description'] = df['description'].apply(process_description)
    
    # Rimuovi le righe con titoli completamente corrotti.
    df = df[df['title'] != "Title Not Available"]

    def normalize_title(title):
        # Verifica che il titolo sia una stringa valida
        if not isinstance(title, str):
            print(f"Titolo non valido: {title}")
            return None  # Indica che il titolo è invalido

        try:
            # 1. Contraction Expansion
            contractions = {
                "I'm": "I am", "you're": "you are", "he's": "he is",
                "she's": "she is", "it's": "it is", "we're": "we are",
                "they're": "they are", "can't": "cannot", "won't": "will not",
                "don't": "do not", "didn't": "did not", "isn't": "is not",
            }
            for contraction, full_form in contractions.items():
                title = re.sub(r'\b' + contraction + r'\b', full_form, title)

            # 2. Tokenizzazione
            tokens = word_tokenize(title)

            # 3. Conversione dei numeri in parole
            tokens = [
                num2words(word) if word.isdigit() else word
                for word in tokens
            ]

            # 4. Lemmatizzazione
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word]

            # 5. Trasformazione in Minuscolo
            tokens = [word.lower() for word in tokens]

            # 6. Stopword Removal
            tokens = [word for word in tokens if word not in stop_words]

            # Ricostruisce il titolo normalizzato
            normalized_title = ' '.join(tokens)

            # Verifica che il titolo normalizzato non sia vuoto
            return normalized_title if normalized_title.strip() else None
        except Exception as e:
            print(f"Errore durante la normalizzazione del titolo: {title}. Errore: {e}")
            return None
        
    df['title'] = df['title'].apply(normalize_title)
    
    # Verifica finale del processo di pulizia.
    df.reset_index(drop=True, inplace=True)
    print("< Data cleaning completato con successo! >")
    return df
