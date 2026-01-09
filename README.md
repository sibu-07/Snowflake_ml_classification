# Snowflake ML Classification

This project demonstrates a machine learning classification workflow using data from Snowflake. It fetches data, performs exploratory data analysis (EDA), and trains multiple classification models to predict a diagnosis.

## Prerequisites

- Python 3.8+
- Snowflake Account

## Installation

1.  Clone the repository.
2.  Install the required packages:
    ```bash
    pip install snowflake-connector-python pandas numpy matplotlib seaborn scikit-learn python-dotenv
    ```

## Configuration

1.  Create a `.env` file in the root directory.
2.  Add your Snowflake credentials to the `.env` file:

    ```env
    SNOWFLAKE_USER=your_username
    SNOWFLAKE_PASSWORD=your_password
    SNOWFLAKE_ACCOUNT=your_account
    SNOWFLAKE_DATABASE=your_database
    SNOWFLAKE_SCHEMA=your_schema
    SNOWFLAKE_WAREHOUSE=your_warehouse
    ```

## Usage

Run the script:

```bash
python snowflake_ml_classification.py
```

## Model Performance

| Model | Accuracy |
| :--- | :--- |
| Random Forest | 0.964912 |
| Logistic Regression | 0.956140 |
| Decision Tree | 0.947368 |
| K-Nearest Neighbors | 0.754386 |
| Support Vector Machine | 0.622807 |
| Gaussian Naive Bayes | 0.614035 |
