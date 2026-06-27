Major projecr : This project aims to detect the patient mental health and depression by monitoring their social media activity

Project Title :
PersonaSense: AI-Based User Behavior Intelligence System 

Project Sub-title : 
An Intelligent Framework for Depression Risk Assessment Through Social Media and Behavioral Analysis 

Model Name : 
SABER (Social Media and Affective BEhavior Recognition) 

Dataset used : 
Depression Dataset: https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset 
Emotion Dataset : https://www.kaggle.com/datasets/debarshichanda/goemotions?resource=download 


Dataset used : 

reddit_depression_dataset.csv 
Shape: (2470778, 8) 
Columns: Index(['Unnamed: 0', 'subreddit', 'title', 'body', 'upvotes', 'created_utc', 'num_comments', 'label']) 

 
Not Depressed (0) = 19,90,261 
Depressed (1) = 4,80,411 

 
saber_dataset_v1.csv  
Shape: (2470778, 2) 
Columns : ['title', 'body'] 


saber_mini_dataset.csv 
Mini Dataset Shape: (100000, 2) 
Columns : ['title', 'body'] 

Label Distribution: 
0.0  -  50000 
1.0  -  50000 

in our SABER version 1 we work on mini dataset as it easy and balanced structure
 
