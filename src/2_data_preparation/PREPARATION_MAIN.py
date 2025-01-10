import data_cleaning
import feature_selection
import feature_scaling
import data_balancing
import pandas as pd

if __name__ == "__main__":
    # Carica il dataset
    print("1. Caricamento dataset con rumori...")
    dataset_path = "../0_data/dataset_enhanced.csv"
    df = pd.read_csv(dataset_path)
    
    # Esegui il data cleaning
    cleaned_df = data_cleaning.clean_dataset(df)
    
    # Salva il dataset pulito
    cleaned_df.to_csv("../0_data/dataset_clean.csv", index=False)
    print("1. Dataset pulito salvato con successo!")
    
    # Carica il dataset
    print("2. Caricamento dataset con Data Clean...")
    dataset_path = "../0_data/dataset_clean.csv"
    df = pd.read_csv(dataset_path)
    
    # Applica la normalizzazione del testo
    for col in ['likes', 'dislikes', 'comments']:
        scaled_df = feature_scaling.scale_column(df, col)
    
    # Salva il dataset aggiornato
    scaled_df.to_csv('../0_data/dataset_scaling.csv', index=False)
    print("2. Dataset con Feature Scaling salvato con successo!")
    
    # Carica il dataset
    print("3. Caricamento dataset con Feature Scaling...")
    dataset_path = "../0_data/dataset_scaling.csv"
    df = pd.read_csv(dataset_path)
    
    # Esegui il Feature Selection
    selection_df = feature_selection.feature_selection(df)

    # Salva il dataset
    selection_df.to_csv("../0_data/dataset_select.csv", index=False)
    print("3. Dataset con Feature Selection salvato con successo!")

    # Carica il dataset
    print("4. Caricamento dataset con Feature Selection...")
    dataset_path = "../0_data/dataset_select.csv"
    df = pd.read_csv(dataset_path)
    
    # Visualizza la distribuzione delle etichette
    data_balancing.plot_label_counts(df)

    print("4. Data Balancing phase visualizzata!")