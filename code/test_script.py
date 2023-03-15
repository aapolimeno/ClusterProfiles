#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:39:17 2023

@author: 5610710
"""

import dask 
import numpy as np
import dask.dataframe as dd 
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, TFAutoModel


# Article data

ddf_articles = dd.read_csv("../data/sample_articles.csv")
#ddf_articles.head(2)

# Select relevant columns
ddf_arts_short = ddf_articles[["title", "num_words", "categories_generated", "keywords_curated"]]
ddf_arts_short.head()

ddf_views = dd.read_csv("../data/interaction_data.csv", assume_missing = True)
#ddf_views


# Interaction data
#ddf_views = dd.read_csv("../data/views_sample.csv", assume_missing = True)
ddf_views = dd.read_csv("../data/interaction_data.csv", assume_missing = True, sep = ";")
ddf_views['ARTICLE_ID'] = ddf_views['ARTICLE_ID'].astype(int)
ddf_views = ddf_views.drop("GEO_ZIPCODE", axis = 1)
#ddf_views = ddf_views.drop("article_id", axis = 1)


# Userneeds data
ddf_needs = dd.read_csv("../Data/userneeds_sample.csv", sep = ";")
ddf_needs = ddf_needs.drop("file_name", axis = 1)

#ddf_needs.head()


article_ids = ddf_needs["ARTICLE_ID"]
ddf_arts_short["ARTICLE_ID"] = article_ids
ddf_arts_short.head()

ddf_needs = ddf_needs.drop("ARTICLE_ID", axis = 1)



# Merge dataframes 

# Merge the two dataframes on the 'article_id' column
merged_df = dd.merge(ddf_views, ddf_arts_short, on='ARTICLE_ID')
# compute the result
result1 = merged_df.compute()

# Repeat to include the userneeds dataframe
merged_df2 = dd.merge(result1, ddf_needs, on="QUASI_USER_ID")
df = merged_df2.compute()

# From now on, continue to work with this dataframe, it contains all relevant columns
#df



def make_mapping(df, variable):
    """
    You can use this function to map variables in a dataframe to a number. Just feed the 
    dataframe and column name into this function. 

    :input df:        Dataframe 
    :input variable:  String, name of the column you want to map to numbers
    :output df :      Dataframe
    
    """
    
    column_list = df[variable]
    
    # Get the unique list values while preserving the input order 
    unique_values = list(OrderedDict.fromkeys(column_list))
    
    # Loop over the unique values and add the mappings to a lookup dictionary
    mapping = {}
    for i, s in enumerate(unique_values):
        mapping[s] = i
        
    # Replace values using the lookup dictionary
    df = df.replace({variable: mapping})
    
    return df



# Apply mappings 

df = make_mapping(df, "GEO_COUNTRY")
df = make_mapping(df, "APP_ID")
#df = make_mapping(df, "REFR_MEDIUM")
df = make_mapping(df, "QUASI_USER_ID")
df = make_mapping(df, "SE_LABEL")
df = make_mapping(df, "SE_ACTION") 

#df.head()


def make_feature_vector(df):
    """
    input: row of a dataframe 
    
    output: a list containing 1 feature vector
    
    """
    
    # Select the columns that can be directly converted into features 
    
    #user_id = df["QUASI_USER_ID"]
    article_id = int(df["ARTICLE_ID"])
    user_id = int(df["QUASI_USER_ID"])
    hour = int(df["dt_hour"])
    weekday = int(df["dt_weekday"])
    
    login = int(df["IS_LOGGED_IN"])
    num_words = int(df["num_words"]) # In the article data, a lot of articles have 0 in this column. That could skew the results
    
    interaction_labels = int(df["SE_LABEL"])
    interaction_actions = int(df["SE_ACTION"])
    #userneeds = df["userneed_geef_me_context"], df["userneed_help_me"], df["userneed_hou_me_op_de_hoogte"], df['userneed_raak_me_verbind_me'], df['userneed_vermaak_me']
    
    # Make vector 
    #vector = [article_id, hour, weekday, login, num_words, userneeds]
    
    vector = [article_id, user_id, hour, weekday, login, num_words, interaction_labels, interaction_actions]
    
    return vector


# Title: sentence embeddings 
sentences = list(df["title"])


model = SentenceTransformer('all-MiniLM-L6-v2') 
embeddings = model.encode(sentences)

embeddings = model.encode(sentences)
print(embeddings)
