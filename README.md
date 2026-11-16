# Online Shopping Purchase Prediction

## Project Overview
A supervised machine learning project that predicts whether an online
shopper is likely to make a purchase based on their browsing behaviour.

## Problem Statement
Online shopping platforms collect large amounts of visitor activity
data, but identifying which visitors are likely to make a purchase is
a challenging task. This project aims to develop a supervised machine
learning classification system that analyses online shopping behaviour
and predicts whether a visitor will make a purchase during a browsing
session. The prediction can help e-commerce businesses understand
customer behaviour and improve their marketing strategies and customer
engagement.

## Dataset

The project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

- **Dataset file:** `online_shoppers_intention.csv`
- **Number of records:** 12,330 (original dataset)
- **Number of features:** 17
- **Target variable:** `Revenue`

The dataset contains information about visitors' browsing activities, page interactions, session duration, visitor characteristics and other session-related attributes.

The target variable `Revenue` indicates whether a purchase was made during the browsing session.

**Dataset Source:** [UCI Machine Learning Repository – Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online)

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook, Visual Studio Code
- **Version Control:** Git and GitHub

## Project Workflow

1. **Data Collection:** Obtained the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

2. **Data Preprocessing:** Checked for missing values and duplicate records, removed duplicates, and prepared the dataset for analysis.

3. **Exploratory Data Analysis (EDA):** Analysed the dataset to understand visitor behaviour, feature distributions, relationships between variables, and purchase patterns.

4. **Feature Engineering and Preprocessing:** Prepared numerical and categorical features using appropriate transformations and encoding techniques.

5. **Data Splitting:** Split the dataset into training, validation, and testing sets to train and evaluate the model.

6. **Model Training and Selection:** Trained and compared machine learning models and selected Random Forest as the final model.

7. **Hyperparameter Tuning:** Used RandomizedSearchCV to explore suitable hyperparameter combinations and improve model performance.

8. **Model Evaluation:** Evaluated the final model using accuracy, precision, recall, F1-score, and confusion matrix.

9. **Model Saving:** Saved the trained model and preprocessing pipeline using Joblib.

10. **Deployment:** Developed an interactive web application using Streamlit to accept shopping session details and predict purchase likelihood.

## Model Performance

The final Random Forest model was evaluated on the independent test dataset.

### Evaluation Metrics
- **Accuracy:** 88.53%
- **Precision:** 60.41%
- **Recall:** 77.49%
- **F1-Score:** 67.89%

The model achieved a recall of 77.49%, indicating that it correctly identified approximately 77% of the actual purchase sessions.
The results show that the model can identify potential purchases while maintaining a balance between precision and recall.


## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/dhyankrshna-r4ce/online_shopping_purchase_prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd online_shopping_purchase_prediction
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Application Features

- Interactive web interface built using Streamlit.
- Input fields for all 17 shopping session features.
- Processes user inputs using the saved preprocessing pipeline.
- Predicts whether a shopping session is likely to result in a purchase.
- Displays clear prediction results: Likely or Unlikely.

## Project Structure

```text
online_shopping_purchase_prediction/
│
├── app.py                                      -> Streamlit web application
├── final_rf_model.pkl                          -> Saved trained Random Forest model
├── preprocessor.pkl                            -> Saved preprocessing pipeline
├── online_shoppers_intention.csv               -> Original dataset
├── online_shopping_purchase_intention.ipynb    -> Jupyter Notebook containing model development
├── requirements.txt                            -> Required Python libraries
├── README.md                                   -> Project documentation
└── .gitignore                                  -> Specifies files Git should ignore
```

## Contact

**GitHub:** [dhyankrshna-r4ce](https://github.com/dhyankrshna-r4ce)

**LinkedIn:** [Dhyan Krishna S](https://www.linkedin.com/in/dhyan-krishna-s)

**Email:** dhyyankrshna@gmail.com