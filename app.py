import streamlit as st
import pandas as pd
import boto3
import os
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False

st.title("ML Project: S3 Dataset Runner")

# AWS Credentials (use environment variables or Streamlit secrets in production)
# AWS Credentials are loaded from environment variables for security.
# Make sure to set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION in your environment.
# Never share your credentials in code or public repositories.
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# Allow user to select AWS region (default from environment or 'eu-west-1')
default_region = os.getenv("AWS_REGION", "")
region_options = ["", "eu-west-1", "us-east-1"]  # Blank option first
AWS_REGION = st.selectbox("Select AWS Region:", region_options, index=region_options.index(default_region) if default_region in region_options else 0)

bucket_name = ""
object_key = ""

# Show bucket selectbox for both 'us-east-1' and 'eu-west-1'
bucket_name = ""
object_key = ""
bucket_options = [""]
if AWS_REGION == "us-east-1":
    bucket_options.append("pythonprojectbuckets3angersm2idee")
bucket_name = st.selectbox("Select your S3 bucket:", bucket_options)
# Only show dataset selectbox if bucket is selected (not blank) and region is 'us-east-1'
if AWS_REGION == "us-east-1" and bucket_name:
    dataset_options = ["", "customer_experience_data.csv"]
    object_key = st.selectbox("Select your dataset file:", dataset_options)

@st.cache_data(show_spinner=True)
def load_data(bucket, key):
    if not bucket or not key:
        return None
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    csv_obj = s3.get_object(Bucket=bucket, Key=key)
    body = csv_obj['Body'].read().decode('utf-8')
    data = pd.read_csv(StringIO(body))
    return data

if bucket_name and object_key:
    data = load_data(bucket_name, object_key)
    if data is not None:
        st.write("## Dataset Preview:")
        st.dataframe(data.head())

        # EDA Section
        with st.expander("📈 Exploratory Data Analysis (EDA)"):
            st.write("Select up to 3 columns for histogram and boxplot, and select pairs for pairplot.")
            numeric_cols = list(data.select_dtypes(include=["number"]).columns)
            string_cols = list(data.select_dtypes(include=["object", "category"]).columns)
            col_choices = numeric_cols + string_cols
            selected_cols = st.multiselect("Choose up to 3 columns for histogram/boxplot:", col_choices, max_selections=3)
            pairplot_cols = st.multiselect("Choose 2 or more columns for pairplot:", numeric_cols, max_selections=3)

            if selected_cols:
                for col in selected_cols:
                    st.write(f"### Histogram for {col}")
                    fig, ax = plt.subplots()
                    if data[col].dtype in ["object", "category"]:
                        data[col].value_counts().plot(kind='bar', ax=ax)
                        ax.set_ylabel("Count")
                    else:
                        sns.histplot(data[col], kde=True, ax=ax)
                    ax.set_xlabel(col)
                    st.pyplot(fig)

                    st.write(f"### Boxplot for {col}")
                    fig2, ax2 = plt.subplots()
                    sns.boxplot(x=data[col], ax=ax2, orient='h')
                    st.pyplot(fig2)

            if pairplot_cols and len(pairplot_cols) >= 2:
                st.write(f"### Pairplot for {', '.join(pairplot_cols)}")
                fig3 = sns.pairplot(data[pairplot_cols])
                st.pyplot(fig3)

        # Show statistics in an expandable section
        with st.expander("📊 Show Basic Statistics"):
            st.write(data.describe())

        # Show columns in an expandable section
        with st.expander("🧩 Show Columns"):
            st.write(list(data.columns))

        # One-hot encoding in an expandable section
        with st.expander("🔢 One-hot Encode String Columns"):
            string_cols = data.select_dtypes(include=["object", "category"]).columns
            if len(string_cols) > 0:
                data_encoded = pd.get_dummies(data, columns=string_cols, drop_first=True)
                st.dataframe(data_encoded.head())
            else:
                st.info("No string columns to encode.")
    else:
        st.warning("Could not load data. Check your bucket/key and AWS credentials.")

    # Machine Learning Section
    with st.expander("🤖 Machine Learning: Predict Retention Status (Churn)"):
        st.write("Train models to predict customer retention (churn) status.")
        if st.button("Run Machine Learning Models"):
            df = data.copy()
            # Prepare features and target
            drop_cols = ["Customer_ID", "Retention_Status", "Retention_Status_Encoded"]
            feature_cols = [col for col in df.columns if col not in drop_cols and not col.endswith("_Encoded")]
            X = df[feature_cols]
            y = df["Retention_Status_Encoded"]
            # Add one-hot encoding for any string columns
            X = pd.get_dummies(X, drop_first=True)
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            def show_metrics(y_true, y_pred, y_proba, model_name):
                acc = accuracy_score(y_true, y_pred)
                ll = log_loss(y_true, y_proba)
                cm = confusion_matrix(y_true, y_pred)
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                st.markdown(f"## {model_name} Results")
                st.success(f"**Accuracy:** {acc:.3f} — This means {acc*100:.1f}% of predictions were correct.")
                st.info(f"**Log Loss:** {ll:.3f} — Lower log loss means better calibrated probabilities. A perfect model has log loss = 0.")
                st.info(f"**ROC-AUC:** {roc_auc:.3f} — Measures the model's ability to distinguish classes. 1 = perfect, 0.5 = random.")
                st.markdown(
                    f"""
                    - **Accuracy** is the proportion of correct predictions (both churned and retained).
                    - **Log Loss** evaluates how close the predicted probabilities are to the true labels.
                    - **ROC-AUC** reflects the model's ability to rank positive cases higher than negative ones.
                    """
                )
                st.markdown("**Classification Report:**")
                st.code(classification_report(y_true, y_pred), language="text")

                st.markdown("**Confusion Matrix:**")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True, linewidths=.5, linecolor='gray')
                # Annotate each cell with white or black text depending on value
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        color = 'white' if cm[i, j] > cm.max()/2 else 'black'
                        ax.text(j+0.5, i+0.5, cm[i, j], ha='center', va='center', color=color, fontsize=16)
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_title('Confusion Matrix', fontsize=14)
                st.pyplot(fig)
                st.caption("The confusion matrix shows the number of correct and incorrect predictions for each class. Diagonal cells are correct predictions; off-diagonal are errors.")

                st.markdown("**ROC Curve:**")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='navy')
                ax2.plot([0, 1], [0, 1], 'k--', label='Random Guess')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('Receiver Operating Characteristic (ROC)')
                ax2.legend(loc='lower right')
                st.pyplot(fig2)
                st.caption("The ROC curve shows the trade-off between sensitivity and specificity. Higher AUC means better discrimination between churned and retained customers.")

            # Logistic Regression
            st.write("### Logistic Regression")
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            y_proba_lr = lr.predict_proba(X_test)[:, 1]
            show_metrics(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")

            # Random Forest
            st.write("### Random Forest Classifier")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            y_proba_rf = rf.predict_proba(X_test)[:, 1]
            show_metrics(y_test, y_pred_rf, y_proba_rf, "Random Forest")

            # XGBoost
            st.write("### XGBoost Classifier")
            if xgb_installed:
                xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                xgb.fit(X_train, y_train)
                y_pred_xgb = xgb.predict(X_test)
                y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
                show_metrics(y_test, y_pred_xgb, y_proba_xgb, "XGBoost")
            else:
                st.warning("XGBoost is not installed. Please install xgboost to use this model.")
else:
    st.info("Please select region, bucket, and dataset.")
