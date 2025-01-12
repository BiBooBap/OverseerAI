<p align="center">
  <img src="./docs/images/logo_overseerai.png" width=200>
  <h1 align="center">OverseerAI</h1>
</p>

This project is aimed to protect users of the web from scams, specifically the ones happening on Youtube.
<br>
The need to solve this problem started when YouTube also started not moderating sponsored content, moreover the ads.
<br>
In fact, many times these types of content totally lack moderation, and most of the times are blatantly scams.
<br>
To solve this issue a Machine Learning Model has been projected, using the CRISP-DM ML Engineering model.
<br>
Further documentation, in italian language, can be found in the root folder of the repository with the name "OverseerAI.pdf" (report).
<br><br>

## Setup

1. Create or activate a Python environment.
2. Install the necessary packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk num2words pyspellchecker joblib
```
<br>

## Steps

### Data Understanding
Go to the "1_data_understanding" folder and run:
```bash
python dataset_enhancer.py
```
This will use the raw dataset "dataset_raw.csv" created synthetically with LLMs, and it will add features, introduce noise, and save the enhanced dataset with the name "dataset_enhanced.csv".
<br><br>

### Data Preparation
Go to "2_data_preparation" and run:
```bash
python PREPARATION_MAIN.py
```
This performs data cleaning, feature scaling, and feature selection, producing the final dataset for modeling as "dataset_select.csv".
<br>
There is also a data balancing step, but it only shows that the dataset is balanced, using matplotlib.
<br><br>

### Data Modeling
Go to "3_data_modeling" and run (for example):
```bash
python train_randomforest.py
```
This trains a Random Forest model, shows evaluation metrics (including confusion matrices), and saves the trained model as "random_forest_model.pkl".
<br><br>

### Deployment
Go to "5_deploy" and run:
```bash
python execute_model.py
```
This executes the saved model, prompting for inputs and displaying the resulting prediction.

