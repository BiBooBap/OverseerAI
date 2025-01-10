import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = "../0_data/dataset_enhanced.csv"
df = pd.read_csv(dataset_path)

def plot_label_counts(df):
    label_counts = df['label'].value_counts()
    total_counts = label_counts.sum()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title('Dataset Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.ylim(0, 250)  # Imposta il limite massimo dell'asse y a 250
    for i, count in enumerate(label_counts.values):
        percentage = (count / total_counts) * 100
        plt.text(i, count + 2, f"{count} ({percentage:.1f}%)", ha='center')
    plt.tight_layout()
    plt.show()

plot_label_counts(df)