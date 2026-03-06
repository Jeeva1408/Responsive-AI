import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.preprocessing import CorrelationRemover
from diffprivlib.models import LogisticRegression as DPLoReg
import shap

np.random.seed(42)
n_samples = 1000

gender = np.random.binomial(1, 0.5, n_samples)
income = np.random.normal(50000, 15000, n_samples)
income = income + (gender * 10000)

logit_approval = (income - 55000) / 15000
prob_approval = 1 / (1 + np.exp(-logit_approval))
loan_status = np.random.binomial(1, prob_approval)

df = pd.DataFrame({
    'gender': gender,
    'income': income,
    'loan_status': loan_status
})

X = df[['gender', 'income']]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cr = CorrelationRemover(sensitive_feature_ids=['gender'])
X_train_fair = cr.fit_transform(X_train)
X_test_fair = cr.transform(X_test)

X_train_fair = pd.DataFrame(X_train_fair, columns=['income_cleansed'])
X_test_fair = pd.DataFrame(X_test_fair, columns=['income_cleansed'])

standard_model = LogisticRegression()
standard_model.fit(X_train, y_train)
standard_preds = standard_model.predict(X_test)

dp_model = DPLoReg(epsilon=1.0, data_norm=100000) 
dp_model.fit(X_train_fair, y_train)
dp_preds = dp_model.predict(X_test_fair)

def print_fairness(y_true, y_pred, sensitive_features, title):
    mf = MetricFrame(
        metrics=selection_rate, 
        y_true=y_true, 
        y_pred=y_pred, 
        sensitive_features=sensitive_features
    )
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Selection Rate by Group:\n{mf.by_group}")
    print(f"Selection Rate Difference: {mf.difference():.3f}")

print_fairness(y_test, standard_preds, X_test['gender'], "Standard Model")
print_fairness(y_test, dp_preds, X_test['gender'], "DP + Fair Model")

explainer = shap.Explainer(standard_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
