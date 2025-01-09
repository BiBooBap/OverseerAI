import pandas as pd

# FEATURE SCALING delle colonne numeriche usando la formula Min-Max.
# Range da 0 a 1.
def scale_column(df, column):
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    df[column] = df[column].round(2)

    return df