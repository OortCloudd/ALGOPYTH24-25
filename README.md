# Streamlit S3 ML Project

https://algopyth24-25.streamlit.app/

This app allows you to:
- Load datasets from AWS S3 buckets
- Explore data with interactive EDA (histograms, boxplots, pairplots)
- Train and evaluate ML models (Logistic Regression, Random Forest, XGBoost) to predict customer churn

This was a nice school project which shows an exemple of a ML cloud deployment in production on the web.

## Setup
1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your AWS credentials as environment variables or in a `.env` file:
   ```ini
   AWS_ACCESS_KEY_ID=your-access-key-id
   AWS_SECRET_ACCESS_KEY=your-secret-access-key
   AWS_REGION=us-east-1
   ```

## Usage
```bash
streamlit run app.py
```

- Select region, bucket, and dataset from the UI.
- Use the EDA section to visualize data.
- Use the ML section to train and evaluate models.

## Notes
- Do NOT commit your real AWS credentials.
- For production, deploy on Streamlit Cloud or AWS EC2.
