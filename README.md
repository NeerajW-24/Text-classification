# 🦐 WSSV Symptom Checker

A Streamlit web app that reads a written description of shrimp symptoms and estimates whether it points to **White Spot Syndrome Virus (WSSV)**. It explains each result by colouring the words that drove the prediction, and double-checks it against a checklist of known WSSV signs.

> This is a screening aid, not a diagnosis. Confirm suspected cases with a laboratory test such as PCR.

## Features

| Tab | What it does |
|---|---|
| 🩺 Diagnose | Type a description (or use an example / pick common signs) to get a verdict, a probability gauge, word-level explanation, a symptom checklist, and what each of the four models says |
| 📋 Batch check | Paste many descriptions or upload a CSV, then download the results |
| 📊 Model performance | Accuracy, precision, recall, F1 and ROC AUC for every model, a confusion matrix, ROC curves, and the sentences the best model got wrong |
| 🗂️ Dataset | Class balance, most distinctive words, a searchable table, and the rows removed during cleaning |
| ℹ️ About | What WSSV is and how the app works |

## How it works

1. **Cleaning**: fixes Windows-encoded apostrophes, removes 2 duplicate rows and 1 row that is not a symptom description (row 5). The original CSV is not changed.
2. **Features**: TF-IDF on words, word pairs and character patterns.
3. **Models**: Linear SVM, Logistic Regression, Naive Bayes and Random Forest.
4. **Testing**: 5-fold cross-validation repeated 10 times. The model with the best F1 score becomes the default (currently Linear SVM, about 97% accuracy).
5. **Explanation**: each word is removed in turn to measure how much it changes the prediction.
6. **Symptom checklist**: expert rules that look for known WSSV signs, including negations such as "no white spots".

The models are trained when the app starts (about 15 seconds on first load, cached afterwards), so there are no model files to keep in sync with library versions.

### Known limitation of the data

The "not WSSV" sentences are general, descriptive facts about shrimp, not observations of healthy ponds or of other diseases. The models therefore partly learn writing style, and a plain sentence such as "shrimp are eating well" can look like WSSV to them. The app flags this case ("Model says WSSV, but no symptoms found"). Adding real healthy-pond and other-disease observations will improve the model.

## What's in this folder

| File / folder | Purpose |
|---|---|
| `app.py` | The website |
| `wssv_model.py` | Data cleaning, training, evaluation, explanation and the symptom checklist |
| `data/wssv_symptoms.csv` | The labelled dataset (`message`, `category` = yes/no) |
| `requirements.txt` | Exact library versions |
| `.python-version` | Python version (3.12) |
| `.streamlit/config.toml` | Theme settings |
| `run_app.bat` / `run_app.sh` | One-click launchers (Windows / Mac-Linux) |

## Run it on your own computer

**Requirement:** Python 3.12 (3.11 or 3.13 should also work). Download it from python.org and tick "Add Python to PATH" during install.

- Windows: double-click `run_app.bat`
- Mac/Linux: run `./run_app.sh` in a terminal

The first run installs the libraries (a few minutes). After that the app opens at <http://localhost:8501>.

**Manual way**
```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Host it online for free (Streamlit Community Cloud)

1. Upload this folder to a GitHub repository (the `.gitignore` already skips `.venv/`).
2. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app**.
3. Choose the repository, branch `main`, main file `app.py`.
4. Under **Advanced settings**, set Python version to 3.12, then **Deploy**.

## Adding more diseases later

Put the new labelled sentences in `data/` and extend `wssv_model.load_dataset` to read a disease name per row instead of yes/no. The models used here all support multiple classes.

## Troubleshooting

| Problem | Fix |
|---|---|
| Port already in use | `streamlit run app.py --server.port 8502` |
| `Dataset not found` | `data/wssv_symptoms.csv` must sit in the `data` folder next to `app.py` |
| Fonts look plain | Fonts load from Google Fonts, so an internet connection is needed. The app still works without one |
