import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import joblib
import matplotlib.pyplot as plt

# Caricamento del dataset
def load_dataset(filepath):
    return pd.read_csv(filepath)

# Creazione della pipeline di preprocessamento e classificazione
def create_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('title_tfidf', TfidfVectorizer(max_features=5000), 'title'),
            ('description_tfidf', TfidfVectorizer(max_features=5000), 'description'),
            ('link_desc_tfidf', TfidfVectorizer(max_features=5000), 'link_desc')
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
    ])
    return pipeline

# Addestramento e valutazione
def train_and_evaluate(df):
    X = df[['title', 'description', 'link_desc']]
    y = df['label']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_num = 1

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    confusion_matrices = []

    y_true_all = []
    y_pred_all = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Fold {fold_num} - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1-Score: {f1:.3f}")

        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)

        fold_num += 1

    fig_cm_folds, ax_cm_folds = plt.subplots(figsize=(8, 4))
    ax_cm_folds.axis('off')
    table_data_cm = []
    for i, cm in enumerate(confusion_matrices, start=1):
        cm_list = cm.tolist()
        flatten = [str(item) for row in cm_list for item in row]
        table_data_cm.append([f"Fold {i}"] + flatten)
    num_values = len(table_data_cm[0]) - 1  # sottraiamo "Fold"
    columns_cm = ["Fold", "TN", "FP", "FN", "TP"]

    tbl_cm = ax_cm_folds.table(cellText=table_data_cm, colLabels=columns_cm, loc='center')
    tbl_cm.auto_set_font_size(False)
    tbl_cm.set_fontsize(8)
    plt.tight_layout()
    plt.show()

    final_cm = confusion_matrix(y_true_all, y_pred_all)

    fig_final_cm, ax_final_cm = plt.subplots(figsize=(8, 2))
    ax_final_cm.axis('off')
    final_cm_list = final_cm.tolist()
    table_data_final_cm = [str(item) for row in final_cm_list for item in row]
    columns_final_cm = ["TN", "FP", "FN", "TP"]
    tbl_final_cm = ax_final_cm.table(cellText=[table_data_final_cm], colLabels=columns_final_cm, loc='center')
    tbl_final_cm.auto_set_font_size(False)
    tbl_final_cm.set_fontsize(10)
    plt.tight_layout()
    plt.show()

    # Salvataggio del modello
    joblib.dump(pipeline, 'random_forest_model.pkl')
    print("Modello salvato come 'random_forest_model.pkl'")

    # Grafico separate per le metriche
    folds = range(1, 11)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    columns = ["Fold", "Accuracy", "Precision", "Recall", "F1"]
    table_data = []
    for i in range(len(accuracy_scores)):
        table_data.append([
            i + 1,
            f"{accuracy_scores[i]:.3f}",
            f"{precision_scores[i]:.3f}",
            f"{recall_scores[i]:.3f}",
            f"{f1_scores[i]:.3f}"
        ])
    tbl = ax.table(cellText=table_data, colLabels=columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
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

    # F1-Score
    plt.figure(figsize=(6, 4))
    plt.plot(folds, f1_scores, marker='o', color='purple')
    plt.title('F1-Score per fold')
    plt.xlabel('Fold')
    plt.ylabel('F1-Score')
    plt.xticks(folds)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tabella globale finale con medie
    global_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    global_precision = sum(precision_scores) / len(precision_scores)
    global_recall = sum(recall_scores) / len(recall_scores)
    global_f1 = sum(f1_scores) / len(f1_scores)

    fig_global, ax_global = plt.subplots(figsize=(4, 2))
    ax_global.axis('off')
    columns_global = ["Accuracy", "Precision", "Recall", "F1"]
    table_data_global = [[
        f"{global_accuracy:.3f}",
        f"{global_precision:.3f}",
        f"{global_recall:.3f}",
        f"{global_f1:.3f}"
    ]]
    tbl_global = ax_global.table(cellText=table_data_global, colLabels=columns_global, loc='center')
    tbl_global.auto_set_font_size(False)
    tbl_global.set_fontsize(10)
    plt.tight_layout()
    plt.show()

# Esecuzione
if __name__ == "__main__":
    dataset_path = "../0_data/dataset_select.csv"  # Modifica con il percorso del tuo dataset
    df = load_dataset(dataset_path)
    train_and_evaluate(df)
