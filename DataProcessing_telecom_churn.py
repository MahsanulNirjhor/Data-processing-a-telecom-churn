# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:24:30 2019

@author: mahsan
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\mahsan\Desktop\mlcourse.ai\data\telecom_churn.csv')

df.head()

df.info()

df.columns

df['Churn'] = df['Churn'].astype('int64')

a=df.describe()
b = df.describe(include=['bool','object'])

df['Churn'].value_counts()

df['Churn'].value_counts(normalize=True)

c = df.sort_values(by='Total day charge', ascending=False) 

d = df.sort_values(by=['Churn','Total day charge'], ascending = [True,False])

df[df['Churn']==1].mean()

df[df['Churn']==0].mean()

df[df['Churn']==0]['Account length'].mean()

df[(df['Churn']==0) & (df['International plan']=='No')]['Total intl minutes'].max()

df.loc[0:5, 'State':'Area code']

df.iloc[0:5, 0:3]

df.apply(np.max)

df.apply(np.min)

df[df['State'].apply(lambda state: state[0] == 'W')].head()

d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)
df.head()

df= df.replace({'Voice mail plan':d})

df.head()

columns_to_show = ['Total day minutes', 'Total eve minutes', 
                   'Total night minutes']

e = df.groupby(['Churn'])[columns_to_show].describe()
f=df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])

df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + \
                     df['Total night charge'] + df['Total intl charge']
df.head()

# pip install seaborn 
import seaborn as sns
sns.countplot(x='International plan', hue='Churn', data=df);

sns.countplot(x='Customer service calls', hue='Churn', data=df);

df['Customer service calls'] > 3

df['Many_service_calls'] = (df['Customer service calls'] > 3).astype('int')

pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True)


























