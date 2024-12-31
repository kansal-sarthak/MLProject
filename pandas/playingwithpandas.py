import pandas as pd
df = pd.read_csv("./datasets/restaurant_data/tips.csv")
# print(df['total_bill'].sum()) #finds the sum of values in column
# print(df['tip'].mean()) #finds the avg of values in column
# print(df['total_bill'].std()) #finds the standard deviation of values in column
# print(df['total_bill'].count()) #counts the number of values in column
# print(df['total_bill'].min()) #finds the minimum value in column
# print(df['tip'].median()) #finds the median value in column
# print(df['tip'].isnull()) #prints each value and determines if it's null
# print(df['total_bill'].unique()) #prints all unique values
# print(df['tip'].tail(5)) #prints the last 5 values of column (in this case)
# df['tip'] = df['tip']+1 #increases all values in tip by 1

print(df.sample(5))
