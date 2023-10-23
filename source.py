#!/usr/bin/env python
# coding: utf-8

# # Cyberbullying📝
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

# In[5]:


import pandas as pd
import opendatasets as od
import numpy as np
import requests
from bs4 import BeautifulSoup


# In[1]:


# Start your code here

tweets_url= "https://www.kaggle.com/datasets/soorajtomar/cyberbullying-tweets"

od.download (tweets_url, data_dir= "./data")


# In[3]:


suicide_url= "https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch"

od.download (suicide_url, data_dir= "./data")


# In[7]:


cdc_url= "https://www.cdc.gov/mmwr/volumes/69/su/su6901a6.htm"
page= requests.get(cdc_url)
soup= BeautifulSoup (page.content, 'html.parser')
tables = soup.find_all('table')

cdc_data_df = pd.read_html (str(tables))

cdc_data_df


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

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

