# write a python program to concatenate multiple dataframes along rows and columns
import pandas as pd
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
concat_df = pd.concat([df1, df2])
print(concat_df)