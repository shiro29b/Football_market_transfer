#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (25,11)


# In[50]:


df = pd.read_csv('top250-00-19.csv')
df.head(5)


# In[51]:


df.isna().sum()


# In[52]:


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# In[53]:


df['diff'] = df['Transfer_fee'] - df['Market_value']
seasons = df.groupby('Season')['diff'].mean().reset_index()
seasons.plot(x='Season', y='diff')
plt.title('average differences across all seasons')
plt.xlabel('seasons')
plt.ylabel('avg. difference of value and price')
plt.show()


# In[54]:


transfer_trend = pd.pivot_table(df, index='Season', values='Market_value', aggfunc=np.max).reset_index()
transfer_trend.plot(x='Season', y='Market_value')
plt.title('transfer_trend')
plt.xlabel('Season')
plt.ylabel('max_transfer_value')
plt.show()


# In[55]:


age_variation = pd.pivot_table(df, index='Age', values='Transfer_fee', aggfunc=np.max).reset_index()
age_variation.plot(x='Age', y='Transfer_fee')
plt.xlabel('Age')
plt.ylabel('transfer_value')
plt.show()


# In[56]:


sns.set_theme(style="whitegrid")
sns.countplot(x='Position', data=df)


# In[57]:


position_df = pd.pivot_table(df, index='Position', values='Market_value', aggfunc=np.sum).reset_index()
sns.barplot(x='Position', y='Market_value', data=position_df)


# In[58]:


avg_fee_pos = pd.pivot_table(df, index='Position', values='Market_value', aggfunc=np.mean).reset_index()
sns.barplot(x='Position', y='Market_value', data=avg_fee_pos)


# In[59]:


max_value_pos = pd.pivot_table(df, index='Position', values='Market_value', aggfunc=np.max).reset_index()
sns.barplot(x='Position', y='Market_value', data=max_value_pos)


# In[60]:


options = ['LaLiga', 'Serie A', 'Premier League', 'Ligue 1', '1.Bundesliga']
top5_league = df[(df['League_from'].isin(options)) & (df['League_to'].isin(options))]
league_from = pd.pivot_table(top5_league, index='League_from', values='Market_value', aggfunc=np.mean).reset_index()
sns.barplot(x='League_from', y='Market_value', data=league_from)


# In[61]:


league_to = pd.pivot_table(top5_league, index='League_to', values='Market_value', aggfunc=np.mean).reset_index()
sns.barplot(x='League_to', y='Market_value', data=league_to)


# In[62]:


league_from_max = pd.pivot_table(top5_league, index='League_from', values='Market_value', aggfunc=np.max).reset_index()
sns.barplot(x='League_from', y='Market_value', data=league_from_max)


# In[63]:


league_to_max = pd.pivot_table(top5_league, index='League_to', values='Market_value', aggfunc=np.mean).reset_index()
sns.barplot(x='League_to', y='Market_value', data=league_to_max)


# In[64]:


df.drop('Name', axis=1, inplace=True)
df


# In[65]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.Position = labelencoder.fit_transform(df['Position'].values.reshape(-1,1))
df.Team_from = labelencoder.fit_transform(df['Team_from'].values.reshape(-1,1))
df.League_from = labelencoder.fit_transform(df['League_from'].values.reshape(-1,1))
df.Team_to = labelencoder.fit_transform(df['Team_to'].values.reshape(-1,1))
df.League_to = labelencoder.fit_transform(df['League_to'].values.reshape(-1,1))


# In[67]:


dependent_variable = df.iloc[:, :6].values
independent_variable = df.iloc[:, 7].values


# In[71]:


from sklearn.model_selection import train_test_split
dependent_variable_train, dependent_variable_test, independent_variable_train, independent_variable_test = train_test_split(dependent_variable, independent_variable, test_size=0.2, random_state=0)


# In[73]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(dependent_variable_train, independent_variable_train)
classifier.score(dependent_variable_train, independent_variable_train)


# In[ ]:




