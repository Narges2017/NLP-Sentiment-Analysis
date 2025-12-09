# Sentiment Analysis – IMDB Movie Reviews (Portfolio Project)

Natural Language Processing (NLP) project to classify movie reviews as positive or negative using machine learning on the IMDB dataset of 50,000 reviews.

## Project overview

- **Problem**: Automatically classify movie reviews as positive or negative based on review text
- **Dataset**: IMDB 50K movie reviews (balanced: 25k positive, 25k negative)
- **Approach**: Text preprocessing + TF-IDF vectorization + Logistic Regression
- **Tools**: Python, pandas, scikit-learn, matplotlib, NLTK/regex

## Dataset

- **Rows**: 50,000 movie reviews
- **Columns**: `review` (text), `sentiment` (positive/negative)
- **Source**: Kaggle – IMDB Dataset of 50K Movie Reviews
- Data files are not stored in this repository; see `data/README.md` for download instructions

## Methodology

### 1. Data preprocessing

- Removed HTML tags (`<br />`) from review text using regex
- Converted all text to lowercase
- Removed punctuation
- Created numeric labels: `sentiment_label` (0 = negative, 1 = positive)

### 2. Exploratory Data Analysis (EDA)

- Checked for missing values: none found
- Verified class balance: exactly 50/50 split (perfectly balanced)
- Analyzed review length distribution: 4–2,470 words, average ~231 words
- Extracted and visualized most common words in positive vs negative reviews (after removing stop words):
  - **Positive**: "great", "love", "best", "good", "story", "characters", "life"
  - **Negative**: "bad", "dont", "just", "plot", "acting", "really"

### 3. Modeling

- **Train/test split**: 80/20 (40,000 train, 10,000 test), stratified by sentiment
- **Pipeline**:
  - `TfidfVectorizer`: Convert text to numerical features (max 5,000 features, English stop words removed)
  - `LogisticRegression`: Binary classifier (max_iter=1000)
- **Hyperparameters**: Default settings (future work: grid search for optimization)

### 4. Evaluation

- **Test accuracy**: **88.79%**
- **Confusion matrix**:
  - True Negatives: 4,388 | False Positives: 612
  - False Negatives: 509 | True Positives: 4,491
- **Classification metrics**:
  - Negative class (0): Precision 0.90, Recall 0.88, F1-score 0.89
  - Positive class (1): Precision 0.88, Recall 0.90, F1-score 0.89
- **Overall F1-score**: 0.89 (balanced performance for both classes)

## Key insights

### Feature importance (model interpretability)

The Logistic Regression model learned interpretable sentiment indicators:

**Top positive words** (highest coefficients):
- "great" (6.92), "excellent" (6.81), "perfect" (5.16), "best" (5.12), "amazing" (4.95), "wonderful" (4.70), "loved" (4.69), "hilarious" (4.45), "favorite" (4.39), "enjoyed" (4.30)

**Top negative words** (lowest coefficients):
- "worst" (-10.05), "waste" (-8.29), "awful" (-7.70), "bad" (-7.12), "boring" (-6.11), "poor" (-5.61), "terrible" (-5.44), "poorly" (-5.28), "worse" (-5.24), "horrible" (-4.95)

These results show the model correctly learned human-interpretable sentiment patterns, not random correlations.

## Business applications

- **Customer feedback analysis**: Automatically classify product, service, or movie reviews at scale
- **Social media monitoring**: Track brand sentiment on Twitter, Facebook, or review platforms in real-time
- **Customer service prioritization**: Flag negative feedback for urgent follow-up
- **Market research**: Aggregate sentiment trends across thousands of reviews to guide product/marketing decisions

## Project structure

NLP_Sentiment_Analysis/
├── data/
│ └── README.md # Dataset source and download instructions
├── notebooks/
│ └── sentiment-analysis-imdb-movie-reviews-portfolio.ipynb
├── reports/
│ └── (optional: saved charts/figures)
├── src/
│ └── (optional: reusable Python modules)
├── .gitignore
├── LICENSE # MIT License
├── README.md # This file
└── requirements.txt # Python dependencies
