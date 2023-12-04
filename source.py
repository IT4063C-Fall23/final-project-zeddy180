#!/usr/bin/env python
# coding: utf-8

# #  Cyberbullying in social media and it's contribution to increased suicide rates among youths📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below --> 
# The problem that I intend to address with this research and analysis is the increase in cyberbullying that has proven to be detrimental, especially to the younger demographic. Due to the serious psychological and emotional effects it can have on victims, which can include despair, anxiety, and in severe cases, suicide,this issue is important. It is imperative to combat cyberbullying in order to create a supportive and safe digital environment as technology develops and more people, particularly teenagers, become active online.

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# -What is the relationship between recorded suicide cases and cyberbullying
# 
# -Is there a trend by age on the observed sources and targets of cyberbullying online 
# 
# -How can the social media platforms limit or get rid of extreme cyberbullying online 
# 
# -Are there any correlations between the countries with most social media users and the number of suicides reported

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# -There is a positive correlation between the increase in suicide rates, and cyberbullying activities online This implies that cyberbullying has has had a significant impact on the rate of suicide over the last 10 years.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# The Data sources I have identified for this project include:
# 
# -Kaggle datasets from suicide-watch for suicide trend from 1985-2016
# -Kaggle datasets for compiled cyberbullying tweets with word scraping
# -Webscrapped suicide demographic and behavioral data from cdc website 
# 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->
# I will use the data from Kaggle to identify the target demographic for the cyberbullying tweets, I then will analyse that data against the suicide data from both Kaggle and cdc to determine the average age group where suicide is more imminent and using the age and gender demographics to find a correlation by linear Regression and visualizations.

# In[129]:


#importing libraries
import pandas as pd
import opendatasets as od
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import joblib
# Python ≥3.10 is required
import sys
assert sys.version_info >= (3, 10)
import statsmodels.api as sm


# Scikit Learn imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

# to make this notebook's output stable across runs
np.random.seed(42)



# In[47]:


from wordcloud import WordCloud 


# In[ ]:


# Data sources

tweets_url= "https://www.kaggle.com/datasets/soorajtomar/cyberbullying-tweets"

od.download (tweets_url, data_dir= "\data")


# In[20]:


tweets_df = pd.read_csv ('./data/cyber/cyberbullying_tweets.csv')

tweets_df.head()


# In[3]:


#Data Source 2
suicide_url= "https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch"

od.download (suicide_url, data_dir= "./data")


# In[49]:


suicide_df = pd.read_csv ('data/suicide.csv')

suicide_df.head(10)




# In[39]:


suicide_df.describe ()


# In[8]:


#Exploring the records that have null values ready for cleaning
suicide_df.isnull().sum()


# In[9]:


suicide_df.hist (bins=50, figsize= (20, 15) )
plt.show


# In[97]:


print(suicide_df.columns.tolist())


# In[10]:


cdc_url= "https://www.cdc.gov/mmwr/volumes/69/su/su6901a6.htm"
page= requests.get(cdc_url)
soup= BeautifulSoup (page.content, 'html.parser')
tables = soup.find_all('table')

cdc_data_df = pd.read_html (str(tables))

cdc_data_df


# ## Exploratory data analysis
# 
# At this stage, the data I have can show me the recorded suicide number by country. The data can be further explored by creating demographic statistics and try to explore if the suicide rate is prevalent to a certain demographic. 
# 
# Most of the HDI data appears to be missing and and since it is least significant to my topic, I will cleanup the data by removing the column from the dataframe.
# 
# I am unable to string the suicide dataset to the other datasets as the dataset does not specify the reason for the suicide and all that can be made at this point are assumptions

# In[21]:


suicide_df.head()


# In[79]:


suicide_df.info()


# In[25]:


#Viewing a histogram to better understand the data
suicide_df.hist (bins=50, figsize= (20, 15) )
plt.show


# In[91]:


#Plotting the scatter plot to see how the different variables withing the data relate to each other
scatter_matrix (suicide_df[['suicides_no', 'population', 'gdp_per_capita ($)'  ]], figsize=(10, 10))
plt.show


# In[87]:


#Plotting a graph showing how the age groups relate to the number of reported suicide cases
suicide_df.plot(kind='scatter', x='age', y='suicides_no', alpha=0.1, figsize=(10, 10))


# In[37]:


#exploring if there exists a correlation between the different columns
suicide_df.year.corr (suicide_df.suicides_no)


# In[55]:


#Sorting the dataframe biggest suicide number from the data set
suicide_df.sort_values (by= "suicides_no", ascending=False).head (10)


# In[57]:


# Group the data by country and sum the suicides number
suicides_by_country = suicide_df.groupby('country')['suicides_no'].sum().reset_index()

# Sort the data by suicides number in descending order
suicides_by_country = suicides_by_country.sort_values('suicides_no', ascending=False)
suicides_by_country


# In[63]:


pip install plotly


# In[64]:


import plotly.express as px


# In[81]:


# Create the map showing the distibution of the suicide reports across the world
fig = px.choropleth(suicides_by_country, locations='country', locationmode='country names', color='suicides_no', title='Countries with the Most Suicides Number')
fig.show()


# In[52]:


#Generating a wordcloud from the Cyberbullying tweets.
# Concatenate the text from all rows into a single string
text = ' '.join(tweets_df['Text'])

# Generate the word cloud
wordcloud = WordCloud().generate(text)

# Display the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ## Machine Learning Implementation Process

# In[92]:


#Dropping all the missing values from the dataframe
suicide_without_na = suicide_df.dropna () #removing empty data from the data frame


# In[29]:


suicide_without_na.isnull().sum() #Print the new data frame with dropped null values


# In[114]:


#Splitting the data set

train_set, test_set = train_test_split (suicide_df, test_size=0.2, random_state=45)
train_set.head(10)


# In[115]:


#separating the predictor from the target
suicide_X = train_set.drop ('suicides_no', axis=1)
suicide_Y = train_set['suicides_no'].copy()


# In[116]:


#Separating the numerical data from the categorical data and imputing the dataframe columns
from sklearn.impute import SimpleImputer

imputer= SimpleImputer (strategy='median')

suicide_num = suicide_X.drop(['country', 'sex', 'age', 'country-year', ' gdp_for_year ($) ', 'generation'],axis=1)

imputer.fit(suicide_num)


# In[117]:


imputer.statistics_


# In[118]:


#Viewing the correlation matrix for the numerical data
suicide_num.corr()


# In[119]:


#Handling Categorical Data
suicide_cat = suicide_X[['country', 'sex', 'age', 'country-year', ' gdp_for_year ($) ', 'generation']] 


# In[120]:


#Transforming the Categorical Data
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder= OrdinalEncoder()
suicide_cat_encoded = ordinal_encoder.fit_transform (suicide_cat)



# In[121]:


suicide_cat_encoded


# In[122]:


#Using OneHotEncoder to encode the categorical data
from sklearn.preprocessing import OneHotEncoder
oneHot_encoder= OneHotEncoder()
suicide_cat_hot = oneHot_encoder.fit_transform(suicide_cat)

suicide_cat_hot.toarray()


# In[123]:


#Using the Pipeline to fit our numerical and categorical transformed data
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
suicide_num_tr = num_pipeline.fit_transform(suicide_num)


# In[124]:


cat_pipeline = Pipeline([
    ('one-hot-encode', OneHotEncoder())
])
suicide_cat_tr = cat_pipeline.fit_transform(suicide_cat)


# In[127]:


#Performing the full Pipeline fitting
from sklearn.compose import ColumnTransformer

num_features = suicide_num.columns
cat_features = suicide_cat.columns

full_pipeline = ColumnTransformer ([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])
clean_suicide = full_pipeline.fit_transform (suicide_X)


# In[131]:


#Linear Regression Analysis
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression
lin_reg.fit (clean_suicide, suicide_Y)


# ## Peer feedback
# 
# Psolademi commented that they like the concept of my project and what you are doing with this, the datasets are also coning along rightly so far. Based on these comments I will keep on working on my project and take into consideration any further feedback.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# 
# -https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016/
# 
# -https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
# 
# -https://medium.com/linkit-intecs/how-to-upload-large-files-to-github-repository-2b1e03723d2
# 
# -https://www.pewresearch.org/search/cyberbullying
# 
# -stackoverflow.com
# 
# -W3schools.com
# 
# -Youtube.com

# # ⚠️ Make sure you run this cell at the end of your notebook before every submission!
# !jupyter nbconvert --to python source.ipynb

# In[30]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

