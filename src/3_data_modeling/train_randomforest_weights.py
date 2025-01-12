import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import FunctionTransformer
import joblib
import matplotlib.pyplot as plt

def load_dataset(filepath):
    return pd.read_csv(filepath)

def multiply_by_0_4(X):
    return X * 0.4

def multiply_by_0_2(X):
    return X * 0.2

# Creazione della pipeline con pesi personalizzati
def create_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('title_tfidf', Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('weight', FunctionTransformer(multiply_by_0_4, validate=False))  # Peso 4/10
            ]), 'title'),
            ('description_tfidf', Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('weight', FunctionTransformer(multiply_by_0_4, validate=False))  # Peso 4/10
            ]), 'description'),
            ('link_desc_tfidf', Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('weight', FunctionTransformer(multiply_by_0_2, validate=False))  # Peso 2/10
            ]), 'link_desc')
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])
    return pipeline

def train_and_evaluate(df):
    X = df[['title', 'description', 'link_desc']]
    y = df['label']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_num = 1

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        print(f"Fold {fold_num} - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")

        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)

        fold_num += 1

    # Salvataggio del modello
    joblib.dump(pipeline, 'random_forest_model_weighted.pkl')
    print("Modello salvato come 'random_forest_model_weighted.pkl'")

    # Grafico separate per le metriche
    folds = range(1, 11)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    columns = ["Fold", "Accuracy", "Precision", "Recall"]
    table_data = []
    for i in range(len(folds)):
        table_data.append([
            folds[i],
            f"{accuracy_scores[i]:.3f}",
            f"{precision_scores[i]:.3f}",
            f"{recall_scores[i]:.3f}"
        ])
    table = ax.table(cellText=table_data, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(folds, accuracy_scores, marker='o', color='blue')
    plt.title('Accuracy per fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(folds)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Precision
    plt.figure(figsize=(6, 4))
    plt.plot(folds, precision_scores, marker='o', color='green')
    plt.title('Precision per fold')
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.xticks(folds)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Recall
    plt.figure(figsize=(6, 4))
    plt.plot(folds, recall_scores, marker='o', color='red')
    plt.title('Recall per fold')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.xticks(folds)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Esecuzione
if __name__ == "__main__":
    dataset_path = "../0_data/dataset_select.csv"  # Modifica con il percorso del tuo dataset
    df = load_dataset(dataset_path)
    train_and_evaluate(df)