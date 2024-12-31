import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport


churners_df = pd.read_csv("../datasets/credit_card_customer_segmentation_data/BankChurners.csv")
           
#print(churners_df.isnull().sum())

#print(churners_df['Gender'])

## EDA
# feature engineering --> transformation, construction, selection, extraction
# trans --> imputation (no missing vals), encoding, scale
profile = ProfileReport(churners_df)
profile.to_file("yourreport.html")


'''
income_order = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K','$120K +']
sns.barplot(data=churners_df, x='Income_Category', y='Credit_Limit', order=income_order)
plt.xlabel('Income')
plt.ylabel('Credit Limit')
plt.title('Bar graph with Income and Credit Limit')
plt.show()
'''

'''
churners_df['Education_Level'].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.show()
'''

'''
sns.boxplot(data=churners_df, x="Total_Revolving_Bal")
plt.show()
'''

## gender emcoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_values = encoder.fit_transform(churners_df[['Gender']])
encoded_churners_df = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out())
churners_df.drop(columns=['Gender'], inplace=True)
churners_df = pd.concat([churners_df, encoded_churners_df], axis = 1)

# handle education
churners_df['Education_Level'] = churners_df['Education_Level'].replace('Unknown', 'Graduate') #replace w other edu ratio later
encoded_values = encoder.fit_transform(churners_df[['Education_Level']])
encoded_churners_df = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out())
churners_df.drop(columns=['Education_Level'], inplace=True)
churners_df = pd.concat([churners_df, encoded_churners_df], axis = 1)
#print(churners_df.info())

# handle marital status
churners_df['Marital_Status'] = churners_df['Marital_Status'].replace('Unknown', 'Married')
encoder = OrdinalEncoder(categories=[["Single", "Married", "Divorced"]])
churners_df['Marital_Status'] = encoder.fit_transform(churners_df[["Marital_Status"]])
#print(churners_df['Marital_Status'])

# handle income category
churners_df['Income_Category'] = churners_df['Income_Category'].replace('Unknown', 'Less than $40K')
encoder = OrdinalEncoder(categories=[['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K','$120K +']])
churners_df['Income_Category'] = encoder.fit_transform(churners_df[["Income_Category"]])
#print(churners_df['Income_Category'])

# handle card category
encoder = OrdinalEncoder(categories=[['Blue', 'Gold', 'Silver', 'Platinum']])
churners_df['Card_Category'] = encoder.fit_transform(churners_df[["Card_Category"]])
#print(churners_df['Card_Category'])

#print(churners_df.info())

# handle card category
# print(churners_df['Attrition_Flag'].unique())
le = LabelEncoder()
le.fit(churners_df['Attrition_Flag'])
churners_df["Attrition_Flag"] = le.transform(churners_df["Attrition_Flag"])
#print(churners_df["Attrition_Flag"])
#print(churners_df['Attrition_Flag'].sample(10))
