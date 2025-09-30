# 🎓 Student Habits and Academic Performance — OOP-Based Analysis

> An object-oriented Python project analyzing how lifestyle habits impact student exam performance.  
> 📘 Covers data cleaning, statistical analysis, visualizations, and predictive modeling.

---

## 📊 Project Overview

This project explores the relationships between **student habits** (such as study time, sleep, social media use) and **academic outcomes**, using clean OOP principles in Python.

The analysis pipeline includes:

- Modular class design for data loading, cleaning, analysis, and visualization
- Insights into how mental health, diet, and sleep affect academic scores
- Predictive modeling using linear regression
- A clear, reproducible reporting structure

---

## 🧠 Learning Objectives

- Apply object-oriented programming (OOP) concepts in Python
- Develop modular, reusable classes
- Perform exploratory and statistical analysis
- Create insightful visualizations
- Build and evaluate a regression model

---

## 📁 Dataset Description

Each record in the dataset represents a student. Key features include:

| Column          | Description                                  |
|------------------|----------------------------------------------|
| `study_time`     | Daily study hours (float)                    |
| `sleep`          | Nightly sleep hours (float)                  |
| `social_media`   | Daily social media use (float)               |
| `diet_quality`   | Self-rated diet (1–5 scale)                  |
| `mental_health`  | Mental health score (1–10 scale)             |
| `final_score`    | Final exam score (percentage, float)         |

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, pickle
- **Interface**: Jupyter Notebook & Python Module

---

## 🔧 Project Phases

### 1. 📥 Data Loading & Preprocessing
- `DataLoader` class loads and validates dataset
- `DataCleaner` checks for:
  - Missing values
  - Duplicates
  - Invalid ranges

### 2. 📐 Statistical Analysis
- `StudentAnalyzer` class performs:
  - Mean/median study time by mental health tier
  - Correlation between sleep and scores
  - Outlier detection in social media use

### 3. 📊 Visualization
- `VisualizationEngine` class generates:
  - Histogram of study time
  - Scatter plot (sleep vs. final_score)
  - Box plots (score by diet_quality)

### 4. 🤖 Predictive Modeling
- `ScorePredictor` class:
  - Trains linear regression model
  - Predicts scores from habits
  - Saves model using pickle

### 5. 📝 Report Generation
- Markdown or PDF report includes:
  - Key visualizations
  - Statistical summaries
  - Interpretation of results

---

## 🧑🏽‍💻 Author

**Sesethu M. Bango**  
*Data Enthusiast & Developer with a passion for human behavior analytics*  
📫 [Connect on LinkedIn]([https://linkedin.com](https://www.linkedin.com/in/sesethu-bango-197856380/)
