# CodeAlpha Data Analytics Internship — Task 4: Sentiment Analysis

![Python Version](https://img.shields.io/badge/Python-3.14%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NLP](https://img.shields.io/badge/NLP-VADER%20%7C%20NLTK-orange.svg)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)

**Author:** Vadhesh ([@vadhesh4-sys](https://github.com/vadhesh4-sys))  
**Internship:** CodeAlpha Data Analytics Internship  
**Task:** Task 4 — Sentiment Analysis  
**Repository Name:** `CodeAlpha_Task4`  

---

## 📌 Project Overview

This repository contains the complete implementation for **CodeAlpha Data Analytics Internship Task 4: Sentiment Analysis**. 

Using a real-world e-commerce dataset of Amazon consumer product reviews, this project establishes an end-to-end sentiment classification pipeline. We combine lexicon-based Natural Language Processing (**VADER**) with supervised machine learning benchmarks (**TF-IDF + Multinomial Logistic Regression**), audit data quality, generate publication-grade exploratory visualizations, and extract actionable business intelligence.

---

## 🎯 Problem Statement & Objectives

While numerical star ratings provide a quick snapshot of customer satisfaction, they fail to reveal the *underlying causes* behind customer delight or frustration. Written reviews contain granular signals—such as hardware bugs, software quirks, battery performance, ergonomics, and cross-platform compatibility.

### Key Objectives:
1. **Data Auditing & Cleaning:** Identify and eliminate web scraper artifacts (e.g., JavaScript snippets, UI truncation labels) and deduplicate review records across multi-variant product entries.
2. **Lexicon-Based Sentiment Analysis:** Deploy **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*) to categorize reviews into **Positive**, **Neutral**, and **Negative** while computing continuous compound polarity scores.
3. **Exploratory Data Analysis (EDA):** Uncover patterns across customer star ratings, identify frequent keywords in positive vs. negative feedback, and analyze sentiment across top Amazon hardware products.
4. **Validation & Proxy Benchmarking:** Evaluate VADER predictions against user star ratings as proxy ground truth and contrast performance against a supervised **TF-IDF + Logistic Regression** model.
5. **Data-Driven Strategy:** Provide actionable product engineering and customer support recommendations based purely on empirical findings.

---

## 📂 Dataset Description & Source

* **Dataset:** Consumer Reviews of Amazon Products (`7817_1.csv`)
* **Source:** Datafiniti / Kaggle Open E-Commerce Database (CC0 / Public Domain)
* **Raw Records:** 1,597 rows $\times$ 27 attributes
* **Cleaned & Deduplicated Records:** 1,054 authentic customer reviews
* **Primary Features:**
  * `reviews.text`: Raw text of the customer review.
  * `reviews.title`: Review headline / title.
  * `reviews.rating`: Customer rating scale (1.0 to 5.0).
  * `reviews.username`: User identifier (used to eliminate duplicate variant records).
  * `name`: Product name (e.g., *Amazon Tap*, *Amazon Fire TV*, *Kindle Paperwhite*).
  * `categories`: Device category and tags.

---

## 🛠️ Technologies & Libraries Used

| Technology | Purpose |
| :--- | :--- |
| **Python 3.14+** | Core programming environment |
| **Pandas** | High-performance data manipulation and cleaning |
| **NumPy** | Numerical computing and vector math |
| **NLTK (VADER)** | Lexicon- and rule-based sentiment intensity analysis |
| **Scikit-Learn** | TF-IDF vectorization, Logistic Regression, classification metrics |
| **Matplotlib** | Static plot generation and customized chart layouts |
| **Seaborn** | Statistical visualizations (KDE distribution, heatmaps) |
| **Jupyter Notebook** | Interactive reproducible analysis workflow |

---

## 🔬 Methodology

```mermaid
flowchart TD
    A["Raw Dataset (1,597 rows)"] --> B["Data Auditing & Cleaning"]
    B -->|Remove JS Scraping Junk & Truncated Text| C["Valid Reviews (1,587 rows)"]
    C -->|Deduplicate Multi-Variant ASINs| D["Authentic Reviews (1,054 rows)"]
    
    D --> E["Lexicon Analysis (VADER)"]
    E -->|Continuous Compound Score -1.0 to +1.0| F["Sentiment Classification"]
    F -->|Compound >= +0.05| G["Positive (86.72%)"]
    F -->|-0.05 < Compound < +0.05| H["Neutral (3.51%)"]
    F -->|Compound <= -0.05| I["Negative (9.77%)"]
    
    D --> J["Proxy Ground Truth (Star Ratings)"]
    J -->|Ratings 4-5: Pos, 3: Neu, 1-2: Neg| K["Benchmark & Evaluation"]
    F --> K
    K --> L["VADER vs Proxy: 81.09% Accuracy"]
    
    D --> M["Supervised ML Baseline"]
    M -->|TF-IDF + Logistic Regression (80/20 Split)| N["ML Test Accuracy: 73.33%"]
```

### 1. Why VADER?
VADER is specifically optimized for informal e-commerce text:
- **Punctuation & Caps:** Weights capitalization (*"GREAT"*) and exclamation marks (*"!!!"*) as emotional amplifiers.
- **Valence Shifting:** Correctly parses negation (*"not good"*) and intensifiers (*"very powerful"*).
- **Zero-Shot:** Does not require pre-labeled training data, avoiding overfitting.

### 2. Standard Thresholds:
$$\text{Class} = 
\begin{cases} 
\textbf{Positive}, & \text{Compound Score } \ge +0.05 \\
\textbf{Neutral}, & -0.05 < \text{Compound Score } < +0.05 \\
\textbf{Negative}, & \text{Compound Score } \le -0.05 
\end{cases}$$

---

## 📊 Key Findings & Results

### 1. Overall Sentiment Breakdown (1,054 Unique Reviews)

| Sentiment Class | Review Count | Percentage | Average Compound Score | Median Compound Score |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 **Positive** | **914** | **86.72%** | $+0.771$ | $+0.844$ |
| 🔴 **Negative** | **103** | **9.77%** | $-0.432$ | $-0.440$ |
| 🟡 **Neutral** | **37** | **3.51%** | $0.000$ | $0.000$ |
| **Total** | **1,054** | **100.0%** | **$+0.640$** | **$+0.832$** |

### 2. VADER vs. Customer Star Rating Proxy Evaluation (899 Rated Reviews)

Evaluating VADER against user star ratings (Ratings 4–5 = Positive, 3 = Neutral, 1–2 = Negative):

* **Overall Accuracy:** **81.09%**
* **Positive Class:** Precision = **0.89**, Recall = **0.92**, F1-Score = **0.90** (Support: 757)
* **Negative Class:** Precision = **0.41**, Recall = **0.45**, F1-Score = **0.43** (Support: 73)
* **Neutral Class:** Precision = **0.08**, Recall = **0.04**, F1-Score = **0.06** (Support: 69)
* **Weighted Average F1-Score:** **0.80**

### 3. Supervised ML Benchmark (TF-IDF + Logistic Regression)
* **Test Accuracy (80/20 Stratified Split):** **73.33%**
* Confirms that VADER's rule-based lexicon outperforms a basic supervised linear classifier on small, highly imbalanced consumer review datasets without requiring labeled training annotations.

---

## 📈 Visualizations

All visual assets are generated with 300 DPI resolution and saved in the `outputs/` folder:

| Chart | Preview Description |
| :--- | :--- |
| **Overall Sentiment Distribution** | Bar chart showing absolute review frequencies and exact percentages. |
| **Sentiment Share Donut** | Donut chart displaying the proportional share of Positive (86.7%), Negative (9.8%), and Neutral (3.5%) reviews. |
| **Sentiment Across Star Ratings** | Stacked bar chart demonstrating strong concordance between 5-star ratings (94% positive) and 1-star ratings (52% negative). |
| **Compound Score Distribution** | Histogram and KDE curve illustrating the positive skew and distribution across thresholds. |
| **Sentiment by Top Products** | Breakdown comparing feedback on *Amazon Tap*, *Amazon Fire TV*, *Amazon Headphones*, and *Kindle Paperwhite*. |
| **Lexical Word Frequency Comparison** | Horizontal bar plots showing the top informative keywords distinguishing positive vs negative reviews. |
| **Confusion Matrix Heatmap** | Matrix visualizing true vs predicted classes against the rating proxy benchmark. |

---

## 💡 Strategic Business Insights

1. **Hardware Engineering:**
   * **Headphone Controls:** The *Amazon Premium Headphones* accumulated complaints regarding non-functioning volume controls on Apple iOS devices. Packaging and listings should clearly indicate platform compatibility, or firmware should be revised.
   * **Wake-Word Ergonomics:** The *Amazon Tap* received high praise for audio and battery life, but recurring criticism for requiring a manual push-button rather than hands-free Alexa activation.
2. **Customer Experience & Retention:**
   * Automated triage should flag reviews with VADER compound scores $\le -0.50$ directly to Tier-2 support for proactive outreach.
3. **Marketing & Conversion:**
   * Customer delight centered overwhelmingly on "battery life", "sound quality", and "reading comfort". These themes should form the anchor of marketing copy.

---

## 📁 Project Directory Structure

```text
CodeAlpha_Task4/
├── data/
│   └── 7817_1.csv                      # Raw Amazon consumer reviews dataset
├── outputs/
│   ├── compound_score_distribution.png  # VADER compound score distribution
│   ├── confusion_matrix.png             # Heatmap of VADER vs rating proxy
│   ├── sentiment_analysis_results.csv   # Enriched dataset with sentiment scores
│   ├── sentiment_by_product.png         # Product-level sentiment comparison
│   ├── sentiment_by_rating.png          # Stacked bar chart of sentiment across stars
│   ├── sentiment_distribution.png       # Sentiment frequency bar chart
│   ├── sentiment_percentages_donut.png  # Sentiment proportion donut chart
│   └── top_positive_negative_words.png  # Lexical keyword comparison
├── Sentiment_Analysis.ipynb            # Fully executed end-to-end Jupyter Notebook
├── requirements.txt                     # Minimal Python dependencies
├── README.md                            # Comprehensive project report & documentation
└── .gitignore                           # Git ignore rules for clean repository
```

---

## 🚀 How to Run the Project Locally

### Prerequisites
* Python 3.10 or higher (Python 3.14 tested)
* VS Code with the **Python** and **Jupyter** extensions installed

### 1. Clone or Open the Workspace
```bash
cd CodeAlpha_Task4
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
Open VS Code, select your Python kernel (`.venv` or default Python), open `Sentiment_Analysis.ipynb`, and click **Run All**.

Alternatively, execute the notebook from the command line:
```bash
jupyter nbconvert --to notebook --execute --inplace Sentiment_Analysis.ipynb
```

---

## 📤 Git & GitHub Submission Commands

To commit and push this project to your GitHub repository (`CodeAlpha_Task4`):

```bash
# Initialize git repository (if not already initialized)
git init

# Add all project files
git add .

# Create the initial commit
git commit -m "feat: complete CodeAlpha Task 4 sentiment analysis project with verified results"

# Set default branch to main
git branch -M main

# Add your remote repository (replace with your personal repo URL)
git remote add origin https://github.com/vadhesh4-sys/CodeAlpha_Task4.git

# Push to GitHub
git push -u origin main
```

---

## 📜 License
This project is completed as part of the **CodeAlpha Data Analytics Internship Program**. It is open source and available under the [MIT License](LICENSE).
