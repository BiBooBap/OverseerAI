# La pulizia dei dati viene eseguita normalizzando i titoli del dataset.
# Le varie normalizzazioni sono elencate di seguito:
# - Trasformazione in minuscolo
# - Rimozione di tutti i numeri
# - Rimozione di tutti i caratteri speciali
# - Rimozione degli spazi extra

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Data Cleaning
def clean_text(text):
    # Normalizzazione del testo
    text = text.lower()  # Minuscolo
    text = re.sub(r'\d+', '', text)  # Numeri
    text = re.sub(r'[^\w\s]', '', text)  # Caratteri speciali
    text = text.strip()  # Spazi extra
    return text

def data_cleaning(df):
    # La pulizia dei titoli genera un CSV di output i cui titoli sono stati normalizzati, sotto il nome di "cleaned_title".
    df['cleaned_title'] = df['title'].apply(clean_text)
    return df


# 2️⃣ FEATURE SCALING E FEATURE EXTRACTION
def feature_extraction(df):
    """
    Trasforma i titoli puliti in vettori numerici utilizzando TF-IDF.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_title'])
    return X, vectorizer


# 3️⃣ FEATURE SELECTION
def feature_selection(X, y):
    """
    Seleziona le migliori feature usando il test Chi-Squared.
    """
    selector = SelectKBest(score_func=chi2, k=500)  # Seleziona le 500 migliori feature
    X_new = selector.fit_transform(X, y)
    return X_new


# 4️⃣ DATA BALANCING
def data_balancing(X, y):
    """
    Bilancia il dataset usando SMOTE (Synthetic Minority Over-sampling Technique).
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# 5️⃣ PIPELINE COMPLETA
def preprocess_data(input_file, output_file):
    """
    Esegue l'intera pipeline di preprocessing.
    """
    # Carica il dataset
    df = pd.read_csv(input_file)
    
    # Data Cleaning
    df = data_cleaning(df)
    
    # Feature Extraction
    X, vectorizer = feature_extraction(df)
    y = df['label']
    
    # Feature Selection
    X = feature_selection(X, y)
    
    # Data Balancing
    X, y = data_balancing(X, y)
    
    # Salva il dataset preprocessato
    df_resampled = pd.DataFrame(X.toarray())
    df_resampled['label'] = y.reset_index(drop=True)
    df_resampled.to_csv(output_file, index=False)
    
    print(f"✅ Dataset preprocessato salvato in: {output_file}")


# Esegui se il file viene eseguito direttamente
if __name__ == "__main__":
    input_path = 'data/youtube_titles_dataset.csv'
    output_path = 'output/preprocessed_dataset.csv'
    preprocess_data(input_path, output_path)
