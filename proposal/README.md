## Members 
* Nick van der Merwe 
  * S5151332 
  * nick.vandermerwe@griffithuni.edu.au 
* William Dower 
	* S5179910 
	* William.dower@griffithuni.edu.au 
* Ethan Lewis 
	* S5179686 
	* Ethan.lewis2@griffithuni.edu.au 

## Project Title: IMDB Movie Recommender 

## Problem Description 

Movie recommendation has always been a very important area of investigation, especially with the advent of streaming and Netflix. Companies spend lots of money to determine the best way to get subscribers to watch the movies that suit them the best. As such it is a relevant and challenging real-world problem to attempt to create a recommendation system using data collected from thousands of reviews on the website IMBD. Using this data our motivation is to create a movie recommendation system that can recommend movies based on a given profile. 

## Datasets 

The data that we use is collected from IMBD and downloaded from Kaggle. 

These two will need to merge where the id in the group lens references to the Kaggle: 

[https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)

[https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset](https://grouplens.org/datasets/movielens/latest/)

The columns and this merging will be explored in the preliminary report. 

## Algorithms / Techniques 

**FP-Growth** (Ethan): The dataset can be split into groups – for example, all the 1 rating a person gives creates a set, and then find associations using FP-Growth. This will be able to predict if a user gives two things the same rating that they will give another movie the same rating. 

Essentially, FP-Growth functions by creating a frequent pattern tree.  It starts by creating a support count of each item in the data set. It then begins to construct a FP-Tree by creating a branch for each item set, it then builds the tree by connecting item sets which contain the same items. Once the tree is created a conditional tree is then made which will trim the item sets which do not reach the minimum support threshold. Once this is done, we are left with the frequent item set. 

**K Nearest Neighbours** (William): A profile or movie can be selected, and similar profiles or movies can be selected from this so that the users or movies can be recommended to each other. 

Essentially, this would function by measuring the Euclidean distance between profiles or movies based on numerical values or sets (i.e., genres can contribute).  

**Density-based clustering** (Nick): Profiles and movies can be grouped into clusters by density so that people can be put into groups based on their preferences. 

This essentially functions by highlighting areas with high density as a group until it reaches a point where it is not dense. The idea behind this is that users would be separated into their own groups of interests with borders in density. 

## Measurements to Evaluate 

For FP-Growth, the effectiveness of the model will be tested by evaluating the number of frequent patterns found in the dataset. Since, unless the algorithm is brute forcing the dataset (which is highly inefficient), any algorithm will most likely miss some patterns, so minimising missed patterns is the goal. This can be tuned by changing the minimum support count to find the correct balance between number of frequent patterns and quality of them. 

For k nearest neighbours, the train/test/validate method will be used to train and tune the model. In this method, the total dataset is split into 3 progressively smaller sets – the training set, which makes up the data of the k nearest neighbours' model, which it uses to make predictions, the testing set, which is used to test the model and tune its hyper-parameters (such as the k value), and the validation set, which is used to find the final effectiveness of the model. This validation set will be used to generate the evaluation measurement for the algorithm, which is its percentage accuracy, calculated based on the error between its prediction and the true value. 

For density-based clustering, we can take a movie out of a person’s profile and see if the movie recommendation system recommends that same movie with the same or worse score – with this we have the potential to generate an accuracy / precision / recall / F value. However, these will likely be low.  

Furthermore, as there are many rows then n-fold cross-validation can also be used from this, and a train/test/validate method. Other fields that can be explored include robustness (removing movies from a user), scalability, interpretability (give the set of films the enjoyed or the kNN distance).  

## Preliminary 

Visible in preliminary.ipynb or the html format for easy opening. 
