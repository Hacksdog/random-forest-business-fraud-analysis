# Random Forest Business & Fraud Analysis

This repository contains two machine learning projects implemented using the **Random Forest Classification algorithm**.  
The objective is to apply **Exploratory Data Analysis (EDA)** and **Random Forest models** to solve real-world business and fraud-related problems.

---

##  Projects Included

### 1️ Company Sales Analysis
**Objective:**  
To identify the key attributes that influence **high sales** in a cloth manufacturing company.

**Approach:**  
- Sales variable is converted into a categorical target (`High` / `Low`)
- Random Forest Classification is applied
- Feature importance is analyzed to identify major sales drivers

**Key Features:**
- Price
- Shelve Location
- Advertising Budget
- Income
- Competitor Price
- Population

---

### 2️ Fraud Check Classification
**Objective:**  
To classify individuals as **Risky** or **Good** based on taxable income.

**Target Variable Definition:**
- `Risky` → Taxable Income ≤ 30,000
- `Good` → Taxable Income > 30,000

**Approach:**
- Target variable creation
- Encoding categorical variables
- Random Forest model training and evaluation
- Feature importance analysis

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

