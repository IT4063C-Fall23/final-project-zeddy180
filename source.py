#!/usr/bin/env python
# coding: utf-8

# # A Machine Learning Approach to Detecting Cyberbullying in Social Media📝
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

# In[3]:


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



# In[19]:


# Start your code here

tweets_url= "https://www.kaggle.com/datasets/soorajtomar/cyberbullying-tweets"

od.download (tweets_url, data_dir= "\data")


# In[20]:


tweets_df = pd.read_csv ('./data/cyber/cyberbullying_tweets.csv')

tweets_df.head()


# In[3]:


#suicide_url= "https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch"

#od.download (suicide_url, data_dir= "./data")


# In[5]:


suicide_df = pd.read_csv ('data/suicide.csv')

suicide_df.head()




# In[8]:


#Exploring the records that have null values ready for cleaning
suicide_df.isnull().sum()


# In[9]:


suicide_df.hist (bins=50, figsize= (20, 15) )
plt.show


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


# In[25]:


#Viewing a histogram to better understand the data
suicide_df.hist (bins=50, figsize= (20, 15) )
plt.show


# In[24]:


#Plotting the scatter plot to see how the different variables withing the data relate to each other
scatter_matrix (suicide_df)
plt.show


# ## Data Cleaning

# In[27]:


suicide_without_na = suicide_df.dropna () #removing empty data from the data frame


# In[29]:


suicide_without_na.isnull().sum() #Print the new data frame with dropped null values


# ## Machine Learning Plan

# What type of machine learning model are you planning to use?
# 
# I intend on using the regression model in trying to find the best fit line and for the predictive analysis of the discrete data values. For the Cyberbullying tweets, I intend on using the K-Nearest neighbbor algorithm to try and train the model to recognize patterns in words from the word cloud.
# 
# What are the challenges have you identified/are you anticipating in building your machine learning model?
# 
# The challenges are mostly lack of labeled datasets for most of the sources I find, especially for the cyberbullying datasets. The other challenge is hardship in finding the relation between the datasets due to the missing labels and a clustering attribute.
# 
# 
# How are you planning to address these challenges?
# 
# To deal with these challenges, I consider using active-learning, which may include selecting the most useful information from the unlabelled datasets and labelling them for an easy reference and training of the model. I also consider restructuring the datasets to only extract the columns that may be the most useful for this project.

# ## Machine Learning Implementation Process

# - Extract the age column from the suicide data
# -Extract the number of suicide reports from the data
# -Remove all the empty values from the data
# -Find the Mean
# -Impute the empty values with the mean
# -Plot a graph of the age against the number of suicide reported and label the age groups\
# -Compute the RMSE for the number of suicide reported
# -Separate the data into Train and Test data and display the best fit line.
# -Use the pipeline to predict the future trends using the best fit line.

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

# # ⚠️ Make sure you run this cell at the end of your notebook before every submission!
# !jupyter nbconvert --to python source.ipynb

# In[26]:


get_ipython().system('jupyter nbconvert --to python source.ipynb')

