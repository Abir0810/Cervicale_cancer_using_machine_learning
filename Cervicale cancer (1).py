#!/usr/bin/env python
# coding: utf-8

# # Cercival cancer

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_excel(r"C:\Users\MSI\Desktop\Data.xlsx")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[88]:


total_rows = len(df)


# In[89]:


total_rows


# In[90]:


null_counts = df.isnull().sum()


# In[91]:


null_counts


# In[92]:


print(df.head())


# In[95]:


target_distribution = df['hpv'].value_counts()
print("\nDistribution of target variable:")
print(target_distribution)


# In[97]:


target_percentage = df['hpv'].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


# In[98]:


target_distribution = df['smoking'].value_counts()
print("\nDistribution of target variable:")
print(target_distribution)


# In[99]:


target_percentage = df['smoking'].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


# In[100]:


target_distribution = df['isd'].value_counts()
print("\nDistribution of target variable:")
print(target_distribution)


# In[101]:


target_percentage = df['isd'].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


# In[102]:


target_distribution = df['herpes'].value_counts()
print("\nDistribution of target variable:")
print(target_distribution)


# In[103]:


target_percentage = df['herpes'].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


# In[104]:


target_distribution = df['sy'].value_counts()
print("\nDistribution of target variable:")
print(target_distribution)


# In[105]:


target_percentage = df['sy'].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


# In[108]:


target_distribution = df['oc'].value_counts()
print("\nDistribution of target variable:")
print(target_distribution)


# In[109]:


target_percentage = df['oc'].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


# In[110]:


target_distribution = df['cc'].value_counts()
print("\nDistribution of target variable:")
print(target_distribution)


# In[111]:


target_percentage = df['cc'].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


# In[119]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\MSI\Desktop\Data.xlsx" 
df = pd.read_excel(file_path)

print("First 5 rows of the DataFrame:")
print(df.head())


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns)

df_standardized = df.copy()
df_standardized[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nStandardized data:")
print(df_standardized.head())

output_path = 'standardized_data.xlsx'
df_standardized.to_excel(output_path, index=False)

print(f"\nStandardized data saved to '{output_path}'.")


# In[123]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter


file_path = r"C:\Users\MSI\Desktop\Data.xlsx"   
df = pd.read_excel(file_path)


print("First 5 rows of the DataFrame:")
print(df.head())


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns


scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[numeric_columns] = scaler.fit_transform(df[numeric_columns])


print("\nStandardized data:")
print(df_standardized.head())


output_path = 'standardized_data.xlsx'
df_standardized.to_excel(output_path, index=False)
print(f"\nStandardized data saved to '{output_path}'.")


target_column = 'cc'  
target_distribution = df_standardized[target_column].value_counts()

print("\nDistribution of target variable:")
print(target_distribution)


target_percentage = df_standardized[target_column].value_counts(normalize=True) * 100
print("\nPercentage distribution of target variable:")
print(target_percentage)


total_count = len(df_standardized[target_column])
imbalance_threshold = 0.1  

is_imbalanced = any((count / total_count) < imbalance_threshold for count in target_distribution)
if is_imbalanced:
    print("\nThe data is imbalanced.")
else:
    print("\nThe data is not imbalanced.")


# In[6]:


y = df['cc']
x = df.drop(['cc'],axis=1)


# In[7]:


y


# In[8]:


x


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[11]:


X_train.shape


# In[12]:


X_test.shape


# In[13]:


y_train.shape


# In[14]:


y_test.shape


# In[ ]:





# # Logistic Regression

# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


logmodel=LogisticRegression()


# In[17]:


logmodel.fit(X_train,y_train)


# In[18]:


predictions = logmodel.predict(X_test)


# In[19]:


from sklearn.metrics import classification_report


# In[20]:


classification_report(y_test,predictions)


# In[21]:


from sklearn.metrics import confusion_matrix


# In[22]:


confusion_matrix(y_test,predictions)


# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


accuracy_score(y_test,predictions)


# In[25]:


logmodel.score(X_test,y_test)


# In[26]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


predictions = logmodel.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[29]:


from sklearn import metrics


# In[30]:


Accuracy = metrics.accuracy_score(y_test, predictions)


# In[31]:


Precision = metrics.precision_score(y_test, predictions)


# In[32]:


Sensitivity_recall = metrics.recall_score(y_test, predictions)


# In[33]:


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)


# In[34]:


F1_score = metrics.f1_score(y_test, predictions)


# In[35]:


print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:





# # Naive Bayes

# In[36]:


from sklearn.naive_bayes import GaussianNB


# In[37]:


logmodel =GaussianNB()


# In[38]:


logmodel.fit(X_train, y_train)


# In[39]:


predictions = logmodel.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report


# In[41]:


classification_report(y_test,predictions)


# In[42]:


from sklearn.metrics import confusion_matrix


# In[43]:


confusion_matrix(y_test,predictions)


# In[44]:


from sklearn.metrics import accuracy_score


# In[45]:


accuracy_score(y_test,predictions)


# In[46]:


predictions = logmodel.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[47]:


from sklearn import metrics
Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
Sensitivity_recall = metrics.recall_score(y_test, predictions)
Specificity = metrics.recall_score(y_test, predictions, pos_label=0)
F1_score = metrics.f1_score(y_test, predictions)
print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:





# # Decision Tree

# In[48]:


from sklearn import tree


# In[49]:


logmodel = tree.DecisionTreeClassifier()


# In[50]:


logmodel.fit(X_train, y_train)


# In[51]:


predictions = logmodel.predict(X_test)


# In[52]:


from sklearn.metrics import accuracy_score


# In[53]:


accuracy_score(y_test,predictions)


# In[54]:


from sklearn.metrics import classification_report


# In[55]:


classification_report(y_test,predictions)


# In[56]:


from sklearn.metrics import confusion_matrix


# In[57]:


confusion_matrix(y_test,predictions)


# In[58]:


predictions = logmodel.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=logmodel.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logmodel.classes_)
disp.plot()
plt.show()


# In[59]:


from sklearn import metrics
Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
Sensitivity_recall = metrics.recall_score(y_test, predictions)
Specificity = metrics.recall_score(y_test, predictions, pos_label=0)
F1_score = metrics.f1_score(y_test, predictions)
print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:





# # AdaBoostClassifier

# In[60]:


from sklearn.ensemble import AdaBoostClassifier


# In[61]:


from sklearn.datasets import make_classification


# In[62]:


x_train, y_train = make_classification(n_samples=1000, n_features=4,
n_informative=2, n_redundant=0,
random_state=0, shuffle=False)


# In[63]:


clf = AdaBoostClassifier(n_estimators=100, random_state=0)


# In[64]:


clf.fit(x_train, y_train)


# In[65]:


clf.score(x_train, y_train)


# In[66]:


from sklearn.metrics import classification_report


# In[67]:


classification_report(y_test,predictions)


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


confusion_matrix(y_test,predictions)


# In[70]:


from sklearn import metrics
Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
Sensitivity_recall = metrics.recall_score(y_test, predictions)
Specificity = metrics.recall_score(y_test, predictions, pos_label=0)
F1_score = metrics.f1_score(y_test, predictions)
print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:





# # Neural Network

# In[71]:


from sklearn.neural_network import MLPClassifier


# In[72]:


anna = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)


# In[73]:


anna.fit(x_train, y_train)


# In[74]:


anna.score(x_train, y_train)


# In[75]:


from sklearn.metrics import classification_report


# In[76]:


classification_report(y_test,predictions)


# In[77]:


from sklearn.metrics import confusion_matrix


# In[78]:


confusion_matrix(y_test,predictions)


# In[79]:


from sklearn import metrics
Accuracy = metrics.accuracy_score(y_test, predictions)
Precision = metrics.precision_score(y_test, predictions)
Sensitivity_recall = metrics.recall_score(y_test, predictions)
Specificity = metrics.recall_score(y_test, predictions, pos_label=0)
F1_score = metrics.f1_score(y_test, predictions)
print({"Accuracy":Accuracy, 
       "Precision":Precision,
       "Sensitivity_recall":Sensitivity_recall,
       "Specificity":Specificity,
       "F1_score":F1_score})


# In[ ]:





# In[ ]:





# In[80]:


import seaborn as sns
from matplotlib import pyplot as plt 


# In[81]:


sns.heatmap(x, vmin=50, vmax=100)


# In[82]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Smoker ratio')
plt.scatter(df.age,df.smoking,color='black',linewidth=5, linestyle='dotted')


# In[83]:


x


# In[84]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Smoking years Ratio')
plt.scatter(df.age,df.sy,color='black',linewidth=5, linestyle='dotted')


# In[85]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Herpes ratio')
plt.bar(df.age,df.herpes,color='red',linewidth=5, linestyle='dotted')


# In[86]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Depression Ratio')
plt.bar(df.age,df.des,color='blue',linewidth=5, linestyle='dotted')


# In[87]:


plt.xlabel('Age')
plt.ylabel('')
plt.title('Cancer Ratio')
plt.bar(df.age,df.cc,color='blue',linewidth=5, linestyle='dotted')


# In[ ]:





# In[ ]:




