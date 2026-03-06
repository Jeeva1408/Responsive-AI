import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
df = pd.read_csv('credit_data.csv')

protected_attribute = 'sex'
outcome = 'loan_status'
correlation = df.groupby([protected_attribute, 'zip_code']).size().unstack(fill_value=0)
print("--- Zip Code Distribution by Protected Attribute ---")
print(correlation)
X = df.drop(columns=[outcome]) 
y = df[outcome]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\nOverall Accuracy: {metrics.accuracy_score(y_test, y_pred):.2f}")
test_results = X_test.copy()
test_results['actual'] = y_test
test_results['predicted'] = y_pred
stats = test_results.groupby('sex_male')['predicted'].mean()
print("\n--- Selection Rates by Group ---")
print(stats)
di_ratio = stats.min() / stats.max()
print(f"Disparate Impact Ratio: {di_ratio:.2f}")