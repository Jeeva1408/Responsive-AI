import pandas as pd
import re

data = {'Name': ['Alice', 'Bob', 'Charlie'],
'Email': ['alice@email.com', 'bob@work.com', 'charlie@home.com'],
'Feature': [10, 20, 30]}
df pd. DataFrame (data)

def mask_email (email):
return re.sub(r'(?<=.{2}).(?=.*@)', '*', email)
df ['Email'] = df ['Email').apply(mask_email)
print (df)