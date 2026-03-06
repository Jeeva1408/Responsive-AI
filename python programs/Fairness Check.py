import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

X, y = fetch_adult(as_frame=True, return_X_y=True)
y_numeric = (y == '>50K').astype(int)
sensitive_feature = X['sex']
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X_encoded, y_numeric, sensitive_feature, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

gm = MetricFrame(
    metrics=selection_rate,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

print("\n--- Selection Rate by Group (Fairness Check) ---")
print(gm.by_group)

diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)

print(f"\nDemographic Parity Difference: {diff:.4f}")

if diff > 0.1:
    print("Warning: High bias detected. The model favors one group significantly.")
else:
    print("Fairness levels are within an acceptable range.")
