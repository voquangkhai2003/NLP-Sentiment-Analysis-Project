# 🛒 Co.opmart Customer Sentiment Analysis

> End-to-end Vietnamese NLP project — from raw Google Maps reviews to a real-time sentiment monitoring dashboard. Covers data preprocessing, auto-labeling with a pretrained transformer, multi-model ML training, and a deployed Streamlit web app.

---

## 📌 Project Overview

This project analyzes **53,451 customer reviews** scraped from Google Maps across Co.opmart supermarket branches nationwide. The goal is to automatically classify Vietnamese customer feedback as **Positive**, **Neutral**, or **Negative** — and surface insights through an interactive dashboard that supports service quality monitoring and branch performance comparison.

---

## 🗂️ Dataset

| Attribute | Detail |
|---|---|
| **Source** | Google Maps reviews — Co.opmart branches across Vietnam |
| **Total Records** | 53,451 reviews |
| **Languages** | Vietnamese (primary), English, Korean, Russian, Japanese, and more |
| **Rating Distribution** | ⭐×1: 5,838 · ⭐×2: 2,303 · ⭐×3: 8,656 · ⭐×4: 14,058 · ⭐×5: 22,596 |

**Raw columns:** `title` (branch name) · `publishedAtDate` · `originalLanguage` · `text` (original) · `textTranslated` (translated) · `stars`

---

## 🔄 Project Pipeline

```
Raw Reviews (Dataset.csv)
        │
        ▼
1. Text Preprocessing      ← Normalize, expand acronyms, ViTokenizer
        │
        ▼
2. Auto-Labeling           ← Vietnamese-Sentiment-visobert (transformer)
        │
        ▼
3. Feature Extraction      ← TF-IDF (5,000 features, unigrams + bigrams)
        │
        ▼
4. Model Training          ← Logistic Regression / Naive Bayes / SVM / DNN
        │
        ▼
5. Model Export            ← svm_model.pkl + tfidf_vectorizer.pkl
        │
        ▼
6. Streamlit App           ← Real-time sentiment dashboard (app.py)
```

---

## 🧹 Text Preprocessing

Vietnamese text requires domain-specific normalization before modeling:

- **Lowercasing & punctuation removal**
- **Acronym expansion** — 50+ Vietnamese slang/abbreviations mapped to full words (e.g., `ko` → `không`, `sp` → `sản phẩm`, `nv` → `nhân viên`)
- **Word segmentation** — using `pyvi.ViTokenizer` for Vietnamese compound word handling
- **Short comment filtering** — removed reviews with fewer than 3 tokens

---

## 🏷️ Auto-Labeling with Transformer

Since no ground-truth sentiment labels existed, reviews were labeled automatically using a pretrained Vietnamese NLP model:

- **Model:** [`5CD-AI/Vietnamese-Sentiment-visobert`](https://huggingface.co/5CD-AI/Vietnamese-Sentiment-visobert)
- **Mapping:** `POS` → Tích cực · `NEU` → Trung tính · `NEG` → Tiêu cực
- **Batch inference:** 128 reviews per batch using PyTorch

This labeled output (`datasetnew.csv`) serves as the training dataset for downstream ML models.

---

## 🤖 Models Trained

### Machine Learning (TF-IDF features)

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Multinomial Naive Bayes | Probabilistic, fast on sparse features |
| **Linear SVM** ✅ | **Best performer — selected for deployment** |

All models evaluated with `classification_report` (Precision / Recall / F1 per class).

### Deep Learning (Dense Neural Network)
- Architecture: `Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dense(3, Softmax)`
- Optimizer: Adam · Loss: Sparse Categorical Crossentropy · Epochs: 10

---

## 🌐 Streamlit Dashboard (`app.py`)

An interactive web app for real-time sentiment monitoring across Co.opmart branches.

**Features:**
- 📥 **Submit new reviews** — input branch, star rating, and text; model predicts sentiment instantly
- 📊 **Stacked bar chart** — sentiment distribution by branch
- 🏆 **Branch leaderboard** — Top 5 best-performing and Top 5 needing improvement (scored by positive − negative rate)
- 🏬 **Branch drill-down** — detailed sentiment breakdown and latest reviews for any selected branch
- 📈 **KPI cards** — Total Reviews · Positive · Neutral · Negative counts

**Prediction logic:**
- Stars ≤ 2 → Negative (rule-based override)
- Stars ≥ 4 → Positive (rule-based override)
- Stars = 3 → SVM model prediction

---

## 📁 File Structure

```
├── Code_NLP.ipynb           ← Full ML pipeline (EDA, labeling, training, export)
├── app.py                   ← Streamlit dashboard app
├── requirements.txt         ← Python dependencies
│
├── Dataset.csv              ← Raw scraped Google Maps reviews (53,451 rows)
├── datasetnew.csv           ← Preprocessed + auto-labeled dataset
│
├── svm_model.pkl            ← Trained LinearSVC model (joblib)
└── tfidf_vectorizer.pkl     ← Fitted TF-IDF vectorizer (joblib)
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/voquangkhai2003/coopmart-sentiment-analysis.git
cd coopmart-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit
pandas
scikit-learn
joblib
pyvi
plotly
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **Note:** `svm_model.pkl` and `tfidf_vectorizer.pkl` must be in the same directory as `app.py`.  
> To retrain the model from scratch, run all cells in `Code_NLP.ipynb`.

---

## 💡 Key Insights

1. **5-star reviews dominate** (42% of all reviews), but 3-star neutral reviews (16%) represent a significant "at-risk" segment worth monitoring.
2. **Multilingual reviews** (~60% non-Vietnamese) are handled by using pre-translated text, ensuring broader coverage.
3. **SVM outperforms** Logistic Regression and Naive Bayes on imbalanced Vietnamese text due to its margin-based optimization.
4. **Rule-based overrides** for extreme star ratings (≤2 and ≥4) improve real-world prediction reliability by combining signals.
5. **Branch leaderboard** enables HQ teams to quickly identify underperforming locations and prioritize service interventions.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| NLP | `pyvi`, `transformers` (visobert), TF-IDF |
| ML / DL | `scikit-learn`, `TensorFlow/Keras` |
| Visualization | `matplotlib`, `seaborn`, `plotly`, `wordcloud` |
| App | `streamlit` |
| Model Serialization | `joblib` |

---

## 👤 Author

**Vo Quang Khai**
Data Analyst | Finance & Data Science Background
[LinkedIn](https://www.linkedin.com/in/voquangkhaikg2003/) · [GitHub](https://github.com/voquangkhai2003)
