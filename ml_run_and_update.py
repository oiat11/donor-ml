import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pymysql

# --- Load Env ---
load_dotenv()
username = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT", "3306")
database = os.getenv("DB_NAME")

# --- Connect DB ---
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")
print(f"🔌 Connecting to MySQL at {host}:{port} / {database}")

# --- Load Donor Data ---
query = """
SELECT 
    id AS donor_index,
    total_invitations,
    total_attendance,
    last_invitation_attendance,
    invitation_acceptance_rate,
    total_donation_amount,
    largest_gift_amount,
    last_gift_amount,
    attendance_lable,
    donation_lable
FROM Donor
WHERE is_deleted = FALSE
"""
data = pd.read_sql(query, engine)

# --- Attendance Model ---
print("\n===== Attendance Model =====")
attendance_features = [
    'total_invitations',
    'total_attendance',
    'last_invitation_attendance',
    'invitation_acceptance_rate'
]
X_att = data[attendance_features]
y_att = data['attendance_lable']
X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(X_att, y_att, test_size=0.2, random_state=42)

lgbm_att = LGBMClassifier(random_state=42)
lgbm_att.fit(X_train_att, y_train_att)
y_pred_att = lgbm_att.predict(X_test_att)
y_prob_att = lgbm_att.predict_proba(X_test_att)[:, 1]

print(f"Accuracy: {accuracy_score(y_test_att, y_pred_att)}")
print(f"AUC-ROC: {roc_auc_score(y_test_att, y_prob_att)}")
print("Classification Report:\n", classification_report(y_test_att, y_pred_att))

print("\nFeature Importances:")
att_importances = pd.DataFrame({
    'feature': attendance_features,
    'importance': lgbm_att.feature_importances_
}).sort_values(by='importance', ascending=False)
print(att_importances)

# Score for all donors
data['lgbm_attendance_score'] = lgbm_att.predict_proba(X_att)[:, 1]

# --- Donation Model ---
print("\n===== Donation Model =====")
donation_features = [
    'total_donation_amount',
    'largest_gift_amount',
    'last_gift_amount'
]
X_don = data[donation_features]
y_don = data['donation_lable']
X_train_don, X_test_don, y_train_don, y_test_don = train_test_split(X_don, y_don, test_size=0.2, random_state=42)

lgbm_don = LGBMClassifier(random_state=42, learning_rate=0.1, n_estimators=100, reg_lambda=0.1)
lgbm_don.fit(X_train_don, y_train_don)
y_pred_don = lgbm_don.predict(X_test_don)
y_prob_don = lgbm_don.predict_proba(X_test_don)[:, 1]

print(f"Accuracy: {accuracy_score(y_test_don, y_pred_don)}")
print(f"AUC-ROC: {roc_auc_score(y_test_don, y_prob_don)}")
print("Classification Report:\n", classification_report(y_test_don, y_pred_don))

print("\nFeature Importances:")
don_importances = pd.DataFrame({
    'feature': donation_features,
    'importance': lgbm_don.feature_importances_
}).sort_values(by='importance', ascending=False)
print(don_importances)

# Score for all donors
data['lgbm_donation_score'] = lgbm_don.predict_proba(X_don)[:, 1]

# --- Combined Score ---
data['lgbm_combined_score'] = np.round(
    np.power(data['lgbm_attendance_score'] + data['lgbm_donation_score'], 2.4) * 1000, 0
).astype(int)

print("\n===== Top 10 Donors by Combined Score =====")
top10 = data[['donor_index', 'lgbm_combined_score']].sort_values(by='lgbm_combined_score', ascending=False).head(10)
print(top10)

# --- Write Combined Score Back to DB ---
conn = pymysql.connect(
    host=host, user=username, password=password,
    db=database, charset='utf8mb4', cursorclass=pymysql.cursors.Cursor
)
update_query = "UPDATE Donor SET ml_score = %s WHERE id = %s"
with conn:
    with conn.cursor() as cursor:
        update_data = list(zip(data['lgbm_combined_score'], data['donor_index']))
        cursor.executemany(update_query, update_data)
    conn.commit()

print("✅ ml_score updated in Donor table.")
