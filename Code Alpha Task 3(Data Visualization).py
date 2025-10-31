#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("titanic.csv")

os.makedirs("figures", exist_ok=True)


plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Sex")
plt.tight_layout()
plt.savefig("figures/survival_by_sex.png")


plt.figure(figsize=(6,4))
sns.barplot(x='Pclass',y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.savefig("figures/survival_by_class.png")


plt.figure(figsize=(8,4))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title("Age Distribution of Passengers")
plt.tight_layout()
plt.savefig("figures/age_distribution.png")


plt.figure(figsize=(8,4))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare Distribution by Passenger Class")
plt.tight_layout()
plt.savefig("figures/fare_boxplot.png")


numeric_df = df.select_dtypes(include=['number']).fillna(0)
corr = numeric_df.corr()


plt.figure(figsize=(7,6))
sns.heatmap(corr,annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
sns.pointplot(x='FamilySize', y='Survived', data=df)
plt.title("Survival Rate by Family Size")
plt.tight_layout()
plt.savefig("figures/family_size_survival.png")

print("All figures saved in the 'figures' folder.")


# In[ ]:




