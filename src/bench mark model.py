import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Test the downstream impact of MANHATON vs Standard Imputation
df_clean = pd.read_csv('heart_rl_cleaned.csv')

# Prepare features and target
X = pd.get_dummies(
    df_clean.drop(columns=['HeartDisease']), drop_first=True
).values
y = df_clean['HeartDisease'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

print(f'Downstream XGBoost Accuracy: {accuracy_score(y_test, preds):.4f}')
print(f'Downstream ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}')