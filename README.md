# Student Performance Analysis & Prediction

This project analyzes a dataset of student performance in Mathematics and Portuguese language courses to identify the key factors influencing academic success. It culminates in a predictive model that can estimate a student's final grade (G3), enabling early intervention strategies.

---

## Problem Statement

The primary goal of this analysis is to answer the following questions:

- What are the most significant demographic, social, and academic factors that influence a student's final grade?
- Can we build an accurate predictive model for a student's final grade (G3) to facilitate early intervention?
- What are the performance differences and patterns between students in the Mathematics and Portuguese courses?

---

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

<img width="9600" height="6400" alt="Scatter plots comparing students’ grades G1, G2, and G3 for math and Portuguese subjects, with correlation coefficients showing strong positive relationships. The G2 vs G3 plots for both subjects have the highest correlation (r ≈ 0.90+), indicating G2 is the strongest predictor of final grade (G3)." src="https://github.com/user-attachments/assets/189d25b2-61c0-4a8f-9b9f-17bad8f408b7" />

<p align="center"><em>Figure: Strong correlation between prior grades (G1, G2) and final grade (G3), especially G2 (r ≈ 0.90+).</em></p>

Previous grades (`G1` and `G2`) are overwhelmingly the strongest predictors of the final grade `G3`. The grade from the second period (`G2`) alone contributes to 89% of the final prediction's explanation. 

### Course Performance Varies

<img width="12000" height="2400" alt="Side-by-side histograms showing grade distributions (G1, G2, G3) for math and Portuguese students. Portuguese students consistently score higher and improve over time, while math students’ scores slightly decline." src="https://github.com/user-attachments/assets/c9b73adb-3ba2-4628-9ac2-be038971f818" />

<p align="center"><em>Figure: Distribution of student grades (G1, G2, G3) by subject. Portuguese students tend to outperform math students, with scores improving over time, while math scores show a slight decline.</em></p>

Portuguese students consistently achieve higher average scores than Mathematics students, and their performance trends upwards over the three periods. Conversely, the average score for Mathematics students tends to decline. 

### Failures Create a Negative Cycle

<img width="8000" height="3200" alt="Box plots showing the distribution of student grades (G1, G2, G3) grouped by number of past course failures (0 to 3). As the number of failures increases, the median and overall grade distributions decrease, showing a strong negative impact on performance." src="https://github.com/user-attachments/assets/eee3ffa4-5427-4291-9f44-70148e4a1128" />

<p align="center"><em>Figure: Box plots of G1, G2, and G3 scores grouped by the number of prior course failures. Students with more failures tend to consistently perform worse across all grading periods.</em></p>

The number of past course failures is the most significant non-grade predictor. Students with a history of failure are statistically more likely to have lower grades. 

### Motivation Matters

<img width="8000" height="3200" alt="Violin plots showing the distribution of G1, G2, and G3 scores split by college aspiration (yes vs no). Students who aspire to attend higher education have visibly higher median scores and a tighter distribution compared to those who do not." src="https://github.com/user-attachments/assets/090eba64-315c-4dc8-b1b0-29537b74eece" />

<p align="center"><em>Figure: Violin plots of G1, G2, and G3 scores grouped by students' college aspirations. Those who plan to pursue higher education tend to have higher and more consistent grades across all periods.</em></p>

Students who aspire to pursue higher education consistently achieve higher grades across all periods. 

### Absences Have an Impact

<img width="7200" height="3200" alt="Scatterplot showing G1, G2, and G3 student scores plotted against number of absences. Higher numbers of absences are associated with lower scores, particularly visible in G3. Most high-performing students have fewer than 10 absences." src="https://github.com/user-attachments/assets/11559d53-0247-44a9-a40a-95d9e8d72014" />

<p align="center"><em>Figure: Scatterplot of G1, G2, and G3 scores plotted against number of absences. Students with fewer absences tend to achieve higher scores, indicating that attendance is a strong predictor of academic performance.</em></p>

Student absences (`absences`) is the second most important feature after `G2`, indicating that consistent attendance is crucial for academic success. 

### Top 10 Feature Importance (Gradient Boosting)

<img width="6400" height="4800" alt="Horizontal bar chart ranking the top 10 features influencing final grade predictions. G2 stands out with the highest importance (~0.9), followed by absences and G1 with much smaller contributions. Other features show negligible importance." src="https://github.com/user-attachments/assets/e0faa366-1ada-4699-b5c0-73a04307ecbe" />

<p align="center"><em>Figure: Bar chart showing the top 10 most important features in predicting final grade (G3).</em></p>

The model reveals that G2 (second period grade) is by far the most important feature in predicting the final grade (G3), contributing the majority of the predictive power. While absences and G1 (first period grade) still hold some influence, their impact is significantly smaller. In contrast, non-academic features such as gender, parental education, school support, and commute time have minimal to negligible importance in the model. This suggests that academic performance earlier in the year, especially in the second term, along with consistent attendance, are the most reliable indicators of final success, while demographic or background characteristics contribute very little to grade prediction.

---

## Methodology

1. **Exploratory Data Analysis (EDA):** Visualized distributions and relationships to understand the data. Key insights were drawn from comparing grades between courses, schools, and based on factors like failures and higher education aspirations.
2. **Data Preprocessing:** Checked for and confirmed no missing or duplicate values. Applied One-Hot Encoding to all nominal categorical features. Handled outliers using the IQR method and capping. 40% of the data were identified as outliers before treatment.
3. **Modeling:** Compared 8 different regression models, including Linear models, K-NN, and Ensemble methods (Random Forest, Gradient Boosting, etc.). The Gradient Boosting model was selected for its superior performance across R², MAE, and RMSE metrics.
4. **Optimization:** Performed Hyperparameter Tuning on the Gradient Boosting model using a 5-fold cross-validated Grid Search. Conducted Feature Selection, finding that the top 7 features yielded the optimal R² score of 0.8606.

### Model Performance Comparison

| Model              | R²       | MAE       | RMSE     |
|--------------------|----------|-----------|----------|
| Gradient Boosting  | 0.831268 | 0.905382  | 1.582205 |
| Random Forest      | 0.829237 | 0.918758  | 1.591699 |
| Extra Trees        | 0.816345 | 0.970350  | 1.650691 |
| ElasticNet         | 0.811678 | 1.000336  | 1.671533 |
| XGBoost            | 0.809031 | 0.996032  | 1.683238 |
| Linear Regression  | 0.805333 | 1.069276  | 1.699455 |
| KNN (k=5)          | 0.797339 | 1.070701  | 1.733999 |
| Decision Tree      | 0.691540 | 1.257962  | 2.139260 |

---

## Model Performance

The final, tuned Gradient Boosting model achieved the following performance:

- **Cross-Validation R² Score:** 0.8698 (± 0.0860)
- **Test Set R² Score:** 0.9247
- **MAE:** 0.8748
- **RMSE:** 1.5667

The model's learning curves show it is stable and does not suffer from underfitting, with a slight but acceptable degree of overfitting. Residual analysis confirms that the model's assumptions are largely met. 

### Residual Analysis

<img width="9600" height="3200" alt="Scatter plot comparing actual vs predicted values and residuals from a Gradient Boosting regression model, showing deviation from the ideal fit and non-random residual patterns." src="https://github.com/user-attachments/assets/f40f203b-ea79-4a2c-b3e5-37b94b537a41" />

<p align="center"><em>Figure: Actual vs Predicted and Residual plots of the Gradient Boosting model.</em></p>

The Actual vs Predicted plot shows that the Gradient Boosting model generally captures the trend of the data, but some deviations are visible, especially at lower and mid-range values. The Residual Plot reveals a non-random pattern, indicating potential heteroscedasticity and model bias. This suggests that while the model performs reasonably well, its predictions may be less reliable across certain ranges.

---

## Actionable Recommendations

Based on the model's insights, the following strategies are recommended for early intervention:

- **Prioritize Monitoring of G1 & G2 Scores:** Since past performance is the best predictor, implement a system to flag students whose scores drop significantly or are below a certain threshold (e.g., < 10) for extra classes or consultations.
- **Implement Strict Absence Tracking:** Students with more than 5 days of absence should receive guidance counseling, with increasing parental involvement for more severe cases.
- **Support Students with a History of Failure:** Provide intensive support like structured remedial classes, peer mentoring, and individual counseling for students who have previously failed courses.
- **Boost Student Motivation:** Introduce career guidance programs, professional seminars, and inspiration classes to foster students' aspirations for higher education.
- **Diversify Teaching Methods:** Evaluate and diversify teaching approaches, particularly in Mathematics, to be more engaging (e.g., using gamification, interactive Q&A) to cater to different learning styles.

---

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
- **GitHub:** github.com/afsilmis
