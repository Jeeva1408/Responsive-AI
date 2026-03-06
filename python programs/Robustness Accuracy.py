import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.preprocessing import CorrelationRemover
from diffprivlib.models import LogisticRegression as DPLoReg
import shap

if not hasattr(sklearn.linear_model.LogisticRegression, 'multi_class'):
    setattr(sklearn.linear_model.LogisticRegression, 'multi_class', 'deprecated')

np.random.seed(42)
n_samples = 1000

gender = np.random.binomial(1, 0.5, n_samples)
income = np.random.normal(50000, 15000, n_samples)
income = income + (gender * 10000)

logit_approval = (income - 55000) / 15000
prob_approval = 1 / (1 + np.exp(-logit_approval))
loan_status = np.random.binomial(1, prob_approval)

df = pd.DataFrame({'gender': gender, 'income': income, 'loan_status': loan_status})

X = df[['gender', 'income']]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cr = CorrelationRemover(sensitive_feature_ids=['gender'])
X_train_fair = cr.fit_transform(X_train)
X_test_fair = cr.transform(X_test)

X_train_fair = pd.DataFrame(X_train_fair, columns=['income_cleansed'])
X_test_fair = pd.DataFrame(X_test_fair, columns=['income_cleansed'])

dp_model = DPLoReg(epsilon=1.0, data_norm=100000) 
dp_model.fit(X_train_fair, y_train)
dp_preds = dp_model.predict(X_test_fair)

mf = MetricFrame(
    metrics=selection_rate, 
    y_true=y_test, 
    y_pred=dp_preds, 
    sensitive_features=X_test['gender']
)

print(f"Accuracy: {accuracy_score(y_test, dp_preds):.3f}")
print(f"Selection Rate:\n{mf.by_group}")
print(f"Difference: {mf.difference():.3f}")

explainer = shap.LinearExplainer(dp_model, X_train_fair)
shap_values = explainer.shap_values(X_test_fair)
shap.summary_plot(shap_values, X_test_fair, plot_type="bar")

X_test_corrupted = X_test_fair.copy()
noise = np.random.normal(0, 5000, X_test_corrupted.shape[0])
X_test_corrupted.iloc[:, 0] = X_test_corrupted.iloc[:, 0] + noise

robust_preds = dp_model.predict(X_test_corrupted)

print(f"Robustness Accuracy: {accuracy_score(y_test, robust_preds):.3f}")
