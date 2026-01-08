# ACM Problem Difficulty Prediction

## 1. Project Overview
This project focuses on automating the estimation of difficulty levels for competitive programming problems. In competitive programming, accurately assessing problem difficulty is crucial for creating balanced contests and helping users practice effectively.

Traditionally, difficulty is determined subjectively by problem setters. This project leverages Machine Learning and Natural Language Processing (NLP) to predict difficulty based on the problem's text description, providing a data-driven classification.

### **Files Overview**
* Web UI source code: **app.py**
* Model Evaluation source code: **Python Source Code for model evaluation.py**
* NoteBook of model evaluation: **Jupyter notebook2.ipynb**
* Classificiation model: **clf_model.pkl**
* Regression model: **reg_model.pkl**

## 2. Dataset Used
* **Source:** This project uses TaskComplexity dataset. A novel dataset containing a total of 4,112 programming tasks was created by systematically extracting tasks from various websites using web scraping techniques.
* **Structure:**
  * Title of the problem
  * Problem statement
  * Input/Output description
  * Sample Input / Output
  * Complexity category (Easy / Medium / Hard)
  * Complexity score (0–10 numeric scale)
  * URL for the problem
* **Dataset Link:** https://github.com/AREEG94FAHAD/TaskComplexityEval-24

## 3. Approach and Models Used
The project utilizes a dual approach involving both **Classification** (predicting labels: Easy/Medium/Hard) and **Regression** (predicting specific difficulty scores).

### **Data Preprocessing**
* **Text Cleaning:** Redundant columns (URL, sample I/O, and title) were removed, and 213 rows containing empty data were discarded. The problem, input, and output descriptions were merged into a single text field, LaTeX expressions were converted into plain text to preserve numerical information, and extra whitespaces were stripped.
* **Text Embedding:** Used all-MiniLM-L6-v2 Sentence Transformer for embedding the text into vectors.
* **Feature Engineering** Included text length, mathematical symbol count, number count, and the presence of power notation in the problem statement.

### **Classification**
Logistic Regression and RBF SVM show relatively low test accuracy and F1-scores,
indicating limited ability to model the complexity of the data, despite moderate training
performance. In contrast, the boosting-based models perform better, with LightGBM
achieving the highest test accuracy and F1-score, followed closely by CatBoost. Although
LightGBM and CatBoost have very high training accuracies, the noticeable gap between
training and test performance suggests some degree of overfitting. Overall, ensemble treebased
methods generalize better than linear and kernel-based models for this classification
task.

LightGBM Classifier will be used in our application as it gives best accuracy and F1 score
among these.

### **Regression**
The Ridge Regressor shows moderate performance with similar train and test MAE,
indicating stable but limited modelling capacity for nonlinear relationships. Gradient Boost
Regressor achieves the lowest training error while maintaining competitive test MAE and
RMSE, demonstrating strong generalization compared to the other models. LightGBM also
performs well but shows slightly higher test error and dispersion in predictions.

Overall, Gradient Boosting provides the best balance between accuracy and generalization, and
therefore it is selected as the final regression model for this task.

## 4. Evaluation Metrics
The models were evaluated using standard metrics for classification and regression tasks:

* **Classification Accuracy:** Measures the percentage of correctly predicted difficulty labels (Easy/Medium/Hard).
    * *Achieved Accuracy:* 51.03% for LightGBM (the Classification model used in Auto Judge).
* **F1 Score:** F1 score is the harmonic mean of precision and recall, measuring a model’s balance between false positives and false negatives.
    * *Achieved F1:* 0.46 for LightGBM (the Classification model used in AutoJudge)
* **RMSE (Root Mean Squared Error):** Used for the regression model to measure the average deviation of the predicted difficulty score from the actual score.
    * *Achieved RMSE:* 2.00 for Gradient Boosting Regressor (the Regression model used in AutoJudge).
* **MAE (Mean Average Error):** It is the average of the absolute differences between predicted values and the true values.
    * *Achieved MAE:* 1.68 for Gradient Boosting Regressor (the Regression model used in AutoJudge).

## 5. Steps to Run the Project Locally
Follow these steps to set up and run the project on your machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mshantanusingh/AutoJudge
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd AutoJudge
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Web Interface:**
    ```bash
    streamlit run app.py
    ```

## 6. Explanation of the Web Interface
The project includes a user-friendly web interface built with **Streamlit** to demonstrate the model's capabilities in real-time.

* **Input Section:**
    * Users input a problem description, Input Description and Output Description.
* **Output Section:**
    * **Difficulty Label:** Displays the predicted category (e.g., **"Medium"**).
    * **Difficulty Score:** Displays the estimated difficulty score generated by the regression model out of 10.

## 7. Demo Video
[Link to a 2–3 minute demo video showcasing the Streamlit app and a walkthrough of the code]

## 8. Project Report


## 9. Author
* **Name:** Shantanu Singh
* **Enrollment no.** 23115131
* **Branch:** Electrical Engineering 3rd Year
