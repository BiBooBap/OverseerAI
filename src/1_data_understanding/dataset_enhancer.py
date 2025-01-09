import pandas as pd
import random
import numpy as np
import re
import string

# Caricamento del Dataset Esistente
def load_existing_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Aggiunta di Nuove Feature
def add_features(df):
    # Feature numeriche casuali create partendo dalla prima riga su ogni istanza.
    # Crea una nuova colonna con valori casuali tra 0 e 23
    df['upload_hour'] = np.random.randint(0, 24, df.shape[0])
    # Crea nuove colonne con valori casuali tra 0 e 100000
    df['likes'] = np.random.randint(0, 100001, df.shape[0])
    # Crea nuove colonne con valori casuali tra 0 e 10000
    df['dislikes'] = np.random.randint(0, 10001, df.shape[0])
    # Crea nuove colonne con valori casuali tra 0 e 10000
    df['comments'] = np.random.randint(0, 10001, df.shape[0])

    return df

# Introduzione di Rumore nel dataset.
def add_noise(df):
    def corrupt_title(title):
        # Se il titolo è già mancante, non fare nulla.
        if pd.isnull(title):
            return title

        # Lista dei "metodi di corruzione".
        noise_methods = [
            lambda x: ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(len(x))),  # Sostituzione di caratteri con caratteri casuali
            lambda x: x + ''.join(random.choices(string.punctuation, k=5)),  # Aggiunta di caratteri speciali
            lambda x: ''.join(random.sample(x, len(x))),  # Mescolamento dei caratteri
            lambda x: '###ERROR###'  # Testo generico di errore
        ]

        # Applica un metodo di corruzione casuale.
        return random.choice(noise_methods)(title)

    # Introduci rumore nel 10% dei titoli del dataset.
    noise_indices = df.sample(frac=0.03).index
    df.loc[noise_indices, 'title'] = df.loc[noise_indices, 'title'].apply(corrupt_title)

    # Introduzione di valori mancanti
    for col in ['upload_hour', 'likes', 'dislikes', 'comments']:
        # Estrae casualmente il 10% delle righe e per ognuna imposta i valori delle colonne esplicitate a "NaN".
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    # Introduzione di valori anomali. Le righe selezionate sono il 5% del dataset.
    noise_indices = df.sample(frac=0.05).index
    # Per ogni riga selezionata, imposta i valori delle colonne "likes" e "dislikes" a valori casuali tra 1000000 e 10000000 e tra 500000 e 1000000
    # rispettivamente. La quantita' di numeri generati corrisponde al numero di indici selezionati (`len(noise_indices)`)
    df.loc[noise_indices, 'likes'] = np.random.randint(1000000, 10000000, len(noise_indices))
    df.loc[noise_indices, 'dislikes'] = np.random.randint(500000, 1000000, len(noise_indices))

    return df

# Salvataggio Dataset
def save_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Dataset aggiornato salvato in: {output_path}")

# Funzione Principale (ovviamente)
def main(input_path, output_path):
    print("Caricamento del dataset...")
    df = load_existing_dataset(input_path)

    print("< Aggiunta di features e rumore nei dati numerici >")
    df = add_features(df)

    print("< Introduzione di rumore nei titoli >")
    df = add_noise(df)

    print("< Salvataggio del nuovo dataset >")
    save_dataset(df, output_path)
    print("< Processo completato con successo >")

# Esecuzione diretta
if __name__ == '__main__':
    input_path = "../0_data/dataset_raw.csv"
    output_path = "../0_data/dataset_enhanced.csv"
    main(input_path, output_path)
