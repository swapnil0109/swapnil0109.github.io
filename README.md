## Introduction
Hi, I am Swapnil Chavan, working as a Data Engineer in a leading financial services company in the USA. These are the list of the projects I have implemented during my journey towards a data scientist.
## 1. Predictive Analytics - Portugal Bank Marketing Campaigns 
This project involves the analysis of banking data, specifically the analysis of marketing campaign data provided by a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. In particular, the Portuguese Bank Marketing dataset contains 45211 rows and 16 columns which are a combination of numerical and categorical variables as well as one target variable, Subscription, which represents whether a client will subscribe to a term deposit (yes/no).
### Data Selection
First and foremost thing for any data science project is the data and my project is also not an exception to it. I have used the below criterias for the selection of the data for my term project.

Relevance
Quality
Compatibility
Cost
Ethics
Value
I order to complete this project, the dataset that I have selected is available on Kaggle. It is a dataset describing Portugal bank marketing campaigns results. Conducted campaigns were based mostly on direct phone calls, offering bank client to place a term deposit. If after all marking afforts, client had agreed to place deposit - target variable marked 'yes', otherwise 'no
### Model Selection
The most important steps that comes under the predictive analysis of any dataset is the model selection. It is not sure that each time all the available models will be helpful or will produce the desired outcomes. My intention is to use various model based on the need and/or relevance.

I plan to employ a variety of machine learning models to address our research question effectively. These models include but are not limited to:

a. Linear Regression: For predicting continuous outcomes.

b. Random Forest: To capture complex relationships in the data.

c. Logistic Regression: For binary classification tasks.

d. Neural Networks: To handle complex patterns and relationships.

The choice of these models is based on their suitability for different aspects of our project, including regression, classification, and pattern recognition. The ensemble methods like Random Forest will help us improve predictive accuracy.
## Conclusion
1. As I understand the business reason behind this project, I need to build a predictive model to know if a customer will accept the term deposit offer that the bank is providing.
2. It is important for the bank to know the potential customers and to put an effort in following up with the customers who are more likely to accept the offer.
3. Because the follow-up is through various channels including phone calls, it is more important to not follow up with the customers which are less likely to accept the offer.
4. In this case, the model that I have built has to be more accurate in terms of predicting negatives. True negatives must be more accurate.
5. As my intention is to find "How good the model is predicting true negatives" , I need to choose the model with high specificity. If we compare both the models, It is evident that logistic regression is the best fit in this case
6. Apart from that I also found that logistic regression is better in terms of precision, recall and F1 Score metrics.
## 2. K-Means Explained
K-means clustering is a popular unsupervised machine learning algorithm used to group similar data points into clusters, aiming to minimize the distance between data points and their respective cluster centers (centroids). 
Here's a breakdown of the algorithm:

How it Works:
1. Initialization: Randomly select k data points as initial centroids (cluster centers). 
2. Assignment: Assign each data point to the nearest centroid, forming clusters. 
3. Update: Recalculate the centroids by finding the mean of all data points within each cluster. 
4. Iteration: Repeat steps 2 and 3 until the centroids no longer change significantly or a maximum number of iterations is reached.
