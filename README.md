# Student Performance Analysis & Prediction

This project analyzes a dataset of student performance in Mathematics and Portuguese language courses to identify the key factors influencing academic success. It culminates in a predictive model that can estimate a student's final grade (G3), enabling early intervention strategies.

---

## Problem Statement

The primary goal of this analysis is to answer the following questions:

- What are the most significant demographic, social, and academic factors that influence a student's final grade?
- Can we build an accurate predictive model for a student's final grade (G3) to facilitate early intervention?
- What are the performance differences and patterns between students in the Mathematics and Portuguese courses?

## Data Source

The dataset was obtained from the UCI Machine Learning Repository and combines data from two secondary school student populations in Portugal.

- **Source:** [Link to UCI Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
- **Total Instances:** 1,044 students
    - Mathematics Course: 395 students
    - Portuguese Course: 649 students
- **Features:** The dataset includes demographic, social, and academic features.
- **Target Variable:** `G3` - The final academic grade, on a scale of 0-20.

---

## Data Analysis

### Academic History is Paramount



Previous grades (`G1` and `G2`) are overwhelmingly the strongest predictors of the final grade `G3`. The grade from the second period (`G2`) alone contributes to 89% of the final prediction's explanation. 

### Course Performance Varies



Portuguese students consistently achieve higher average scores than Mathematics students, and their performance trends upwards over the three periods. Conversely, the average score for Mathematics students tends to decline. 

### Failures Create a Negative Cycle


The number of past course failures is the most significant non-grade predictor. Students with a history of failure are statistically more likely to have lower grades. 

### Motivation Matters



Students who aspire to pursue higher education consistently achieve higher grades across all periods. 

**Absences Have an Impact**

Student absences (`absences`) is the second most important feature after `G2`, indicating that consistent attendance is crucial for academic success. 

### Top 10 Feature Importance (Gradient Boosting)



## Methodology

1. **Exploratory Data Analysis (EDA):** Visualized distributions and relationships to understand the data. Key insights were drawn from comparing grades between courses, schools, and based on factors like failures and higher education aspirations.
2. **Data Preprocessing:** Checked for and confirmed no missing or duplicate values. Applied One-Hot Encoding to all nominal categorical features. Handled outliers using the IQR method and capping. 40% of the data were identified as outliers before treatment.
3. **Modeling:** Compared 8 different regression models, including Linear models, K-NN, and Ensemble methods (Random Forest, Gradient Boosting, etc.). The Gradient Boosting model was selected for its superior performance across R², MAE, and RMSE metrics.
4. **Optimization:** Performed Hyperparameter Tuning on the Gradient Boosting model using a 5-fold cross-validated Grid Search. Conducted Feature Selection, finding that the top 7 features yielded the optimal R² score of 0.8606.

**Model Performance Comparison**

`[GRAPH: R2 Score and MAE/RMSE charts from Slide 9]`

## Model Performance

The final, tuned Gradient Boosting model achieved the following performance:

- **Cross-Validation R² Score:** 0.8698 (± 0.0860)
- **Test Set R² Score:** 0.9247
- **MAE:** 0.8748
- **RMSE:** 1.5667

The model's learning curves show it is stable and does not suffer from underfitting, with a slight but acceptable degree of overfitting. Residual analysis confirms that the model's assumptions are largely met. 

**Residual Analysis**

<img width="9600" height="3200" alt="residual" src="https://github.com/user-attachments/assets/f40f203b-ea79-4a2c-b3e5-37b94b537a41" />


## Actionable Recommendations

Based on the model's insights, the following strategies are recommended for early intervention:

- **Prioritize Monitoring of G1 & G2 Scores:** Since past performance is the best predictor, implement a system to flag students whose scores drop significantly or are below a certain threshold (e.g., < 10) for extra classes or consultations.
- **Implement Strict Absence Tracking:** Students with more than 5 days of absence should receive guidance counseling, with increasing parental involvement for more severe cases.
- **Support Students with a History of Failure:** Provide intensive support like structured remedial classes, peer mentoring, and individual counseling for students who have previously failed courses.
- **Boost Student Motivation:** Introduce career guidance programs, professional seminars, and inspiration classes to foster students' aspirations for higher education.
- **Diversify Teaching Methods:** Evaluate and diversify teaching approaches, particularly in Mathematics, to be more engaging (e.g., using gamification, interactive Q&A) to cater to different learning styles.

## Installation & Usage

This application has been deployed via **Streamlit** and is accessible online. No local installation is required for end users.

**Application Link:** [https://student-grade-predictor-dffk3nwfantqx2xb9liygd.streamlit.app](https://student-grade-predictor-dffk3nwfantqx2xb9liygd.streamlit.app/)

To use the application:

1. Open the provided link in your browser.
2. Input the required student data, including:
    - First period and second period grades
    - Age
    - Gender
    - Course
    - Number of absences
    - Social activity level
3. Click the “Predict Final Grade” button.
4. The application will generate:
    - A predicted final grade (e.g., 9/20)
    - Performance classification
    - Comparison to average and passing thresholds
    - Feature impact analysis
    - Recommendations based on model output

---

- **Author:** Az-Zukhrufu Fi Silmi Suwondo
- **Email:** afsilmis@gmail.com
- **GitHub:** [github.com/afsilmis](https://www.google.com/search?q=https://github.com/afsilmis/&authuser=1)
