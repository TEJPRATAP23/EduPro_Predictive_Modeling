# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

## Project Overview

This project develops predictive models to forecast course enrollment demand and course revenue for the EduPro online learning platform.

The objective is to support data-driven decisions related to:

* Course launch planning
* Course pricing
* Instructor onboarding
* Demand assessment
* Revenue forecasting
* Course category analysis

The project follows an end-to-end data analytics and machine learning workflow, covering data preparation, exploratory data analysis, feature engineering, predictive modeling, dashboard development, testing, and final reporting.

## Business Problem

EduPro needs quantitative evidence to understand which courses are likely to generate higher enrollment and revenue.

The project addresses two primary prediction targets:

1. **Enrollment Count**
2. **Course Revenue**

These predictions can support better course planning, pricing decisions, instructor selection, and revenue forecasting.

## Dataset

The project analyzes **60 courses** using course, teacher, and performance-related information.

The modeling dataset contains **9 features**:

* CourseCategory
* CourseType
* CourseLevel
* CoursePrice
* CourseDuration
* CourseRating
* TeacherRating
* YearsOfExperience
* Expertise

Prediction targets:

* EnrollmentCount
* CourseRevenue

Identifier fields such as CourseID, CourseName, TeacherID, and TeacherName are excluded from model training.

## Project Workflow

The project was completed through the following stages:

* **Day 1** — Project Setup
* **Day 2** — Data Cleaning
* **Day 3** — Data Preparation and Merging
* **Day 4** — Course Exploratory Data Analysis
* **Day 5** — Transaction Exploratory Data Analysis
* **Day 6** — Teacher and Course Performance Analysis
* **Day 7** — Feature Engineering
* **Day 8** — Prediction Target Preparation
* **Day 9** — Train/Test Strategy
* **Day 10** — Baseline Models
* **Day 11** — Advanced Models
* **Day 12** — Model Evaluation
* **Day 13** — Feature Engineering
* **Day 14** — Final Model and Prediction Pipeline
* **Day 15** — Streamlit Setup
* **Day 16** — Demand Prediction Dashboard
* **Day 17** — Revenue Dashboard
* **Day 18** — Category Analysis and Feature Importance
* **Day 19** — Dashboard Testing and Error Handling
* **Day 20** — Research Paper and Executive Summary
* **Day 21** — GitHub Project Setup

## Modeling Approach

The project uses machine learning regression techniques to predict enrollment demand and course revenue.

The modeling workflow includes:

1. Data preparation
2. Feature selection
3. Categorical feature encoding
4. Numerical feature processing
5. Train/test splitting
6. Baseline modeling
7. Advanced regression modeling
8. Model evaluation
9. Final model selection
10. Prediction pipeline development

Categorical variables are processed using one-hot encoding, while numerical variables are passed through the modeling pipeline.

## Prediction Targets

### Enrollment Demand

The enrollment model predicts:

`EnrollmentCount`

This provides an estimate of expected course demand.

### Course Revenue

The revenue model predicts:

`CourseRevenue`

This provides an estimate of expected financial performance.

## Train/Test Strategy

The modeling dataset contains **60 course observations**.

A **4:1 train/test split** was used:

* Training observations: **48**
* Testing observations: **12**

The test set was kept separate for model evaluation.

## Dashboard

An interactive **Streamlit dashboard** was developed for course-level prediction and analysis.

The application includes:

* Course input
* Enrollment prediction
* Revenue prediction
* Demand interpretation
* Business recommendations
* Category analysis
* Feature importance
* Input validation
* Error handling

The main application is located at:

`app/app.py`

## Final Models

The trained final models are stored in the `models` directory:

* `EduPro_Final_Enrollment_Model.joblib`
* `EduPro_Final_Revenue_Model.joblib`

## Repository Structure

```text
EduPro_Predictive_Modeling
│
├── data
│   ├── raw
│   ├── cleaned
│   ├── processed
│   ├── feature_data
│   └── model_data
│
├── notebooks
│   ├── Day01_...
│   ├── Day02_...
│   ├── ...
│   ├── Day19_Dashboard_Testing_Error_Handling.ipynb
│   └── Day20_Research_Paper_Executive_Summary.ipynb
│
├── models
│   ├── EduPro_Final_Enrollment_Model.joblib
│   └── EduPro_Final_Revenue_Model.joblib
│
├── app
│   └── app.py
│
├── reports
│   ├── EduPro_Research_Paper.docx
│   ├── EduPro_Research_Paper.pdf
│   ├── EduPro_Executive_Summary.docx
│   └── EduPro_Executive_Summary.pdf
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Running the Dashboard

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application from the project root:

```bash
streamlit run app/app.py
```

The application will open in a web browser.

## Key Project Results

| Metric                |      Result |
| --------------------- | ----------: |
| Courses analyzed      |          60 |
| Modeling features     |           9 |
| Prediction targets    |           2 |
| Training observations |          48 |
| Testing observations  |          12 |
| Demand prediction     | Implemented |
| Revenue prediction    | Implemented |
| Category analysis     | Implemented |
| Feature importance    | Implemented |
| Interactive dashboard | Implemented |

## Reports

The final project documentation is available in the `reports` directory:

* **EduPro Research Paper**
* **EduPro Executive Summary**

Both DOCX and PDF versions are included.

## Limitations

The project uses a relatively small number of course-level observations.

Therefore, model performance should be interpreted as a project-level analytical result rather than a production-scale forecasting guarantee.

Potential future improvements include:

* Larger datasets
* Historical time-series data
* Additional learner behavior variables
* Course engagement metrics
* Marketing information
* More extensive model validation
* Regular model retraining

## Future Scope

Future development could extend the system with:

* Automated model retraining
* Real-time course performance monitoring
* Time-series revenue forecasting
* Learner-level behavioral features
* Advanced hyperparameter optimization
* Model monitoring
* Cloud deployment
* Automated business reporting

## Author

**TEJPRATAP PRAJAPATI**

## Project Title

**Predictive Modeling for Course Demand and Revenue Forecasting on EduPro**
