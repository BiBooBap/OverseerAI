import pandas as pd
import re

def feature_selection(df):
    df_new = df.copy()  # Crea un nuovo DataFrame

    def extract_link(text):
        if pd.isnull(text):
            return "No link"
        match = re.search(r'http[s]?://\S+', text)
        return match.group() if match else "no link"

    # Crea la nuova colonna con i link
    df_new['link_desc'] = df_new['description'].apply(extract_link)

    def remove_link(text):
        if pd.isnull(text):
            return text
        return re.sub(r'http[s]?://\S+', '', text).strip()

    # Rimuove i link dalla colonna 'description'
    df_new['description'] = df_new['description'].apply(remove_link)

    # Pulizia e normalizzazione di link_desc
    def clean_link_desc(link):
        if not isinstance(link, str) or not link:
            return link
        # Elimina i caratteri speciali e inserisce spazi in loro sostituzione
        text = re.sub(r'[^\w\s]', ' ', link)
        # Rimuove spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_new['link_desc'] = df_new['link_desc'].apply(clean_link_desc)

    df_new.drop(['upload_hour','likes','dislikes','comments'], axis=1, inplace=True)
    
    return df_new