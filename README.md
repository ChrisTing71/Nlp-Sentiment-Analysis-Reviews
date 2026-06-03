# Nlp-Sentiment-Analysis-Reviews

An NLP term project for sentiment analysis of customer reviews using classical machine learning.

The project contains a complete Jupyter Notebook workflow and a runnable Python script for training and evaluation. It supports:

- IMDB review sentiment classification.
- Amazon review batch prediction.
- A mixed-domain experiment using half IMDB + half Amazon for training and the remaining halves for testing.
- An interactive demo function for live review prediction during presentations.

## Project Files

- `IMDB_Dataset.csv`: IMDB review dataset with `review` and `sentiment` columns.
- `amazon_reviews.csv`: Amazon review dataset used for cross-domain prediction.
- `sentiment_analysis.ipynb`: Main notebook with step-by-step analysis, visualizations, demo functions, and mixed-domain experiment.
- `sentiment_analysis.py`: Standalone script version of the IMDB training pipeline.

## Requirements

Python 3 with the following packages:

- `pandas`
- `nltk`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install them with:

```bash
pip install pandas nltk scikit-learn matplotlib seaborn
```

## Quick Start

### 1. Open the notebook

Open `sentiment_analysis.ipynb` in VS Code or Jupyter and run the cells from top to bottom.

### 2. Run the script

If you only want the IMDB training pipeline from the command line:

```bash
python sentiment_analysis.py
```

If you use the local virtual environment created for this project:

```bash
source .venv/bin/activate
python sentiment_analysis.py
```

## Notebook Workflow

The notebook is organized into these sections:

1. Import libraries and download NLTK stopwords.
2. Clean the IMDB dataset and convert labels to binary values.
3. Visualize class distribution.
4. Split the IMDB dataset, extract TF-IDF features, and train Logistic Regression.
5. Evaluate the IMDB model with accuracy, classification report, and confusion matrix.
6. Use `predict_my_review(custom_review)` for live demo predictions.
7. Load Amazon reviews, predict sentiment in bulk, and evaluate using `overall`.
8. Plot Amazon prediction distribution, show sample predictions, and display the Amazon confusion matrix.
9. Run the mixed-domain experiment: half IMDB + half Amazon for training, the other halves for testing.

## Amazon Evaluation Rule

For the Amazon dataset, the notebook uses a stricter label rule:

- `4.0` to `5.0` is treated as Positive.
- `1.0` to `2.0` is treated as Negative.
- `3.0` is excluded from evaluation.

This makes the reported metrics more conservative and avoids treating neutral reviews as training targets for evaluation.

## Outputs You Can Expect

- Accuracy score.
- Classification report with precision, recall, and F1-score.
- Confusion matrix plots.
- Bar charts for predicted sentiment distribution.
- Example positive and negative reviews for presentation.

## Notes

- The notebook expects the CSV files to be in the same folder as the notebook.
- The live demo function uses the already trained `vectorizer` and `model` objects.
- The Amazon and mixed-domain sections reuse the same text cleaning logic from `clean_text()`.

## Suggested Presentation Flow

1. Show the IMDB training pipeline and IMDB confusion matrix.
2. Demonstrate `predict_my_review()` with a few positive, negative, and mixed reviews.
3. Show the Amazon batch prediction results and the Amazon confusion matrix.
4. Finish with the mixed-domain experiment for a stronger cross-domain comparison.
