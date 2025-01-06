import pandas as pd

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