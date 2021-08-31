note, this was just lazily slapped in for convenience in the future and is likely not the actual proposal.
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
The goal of this is to make a movie recommendation system for IMDB profiles. For this, we will need a dataset of profiles and what the movies are: 
## Datasets 
These two will need to merge together where the id in the group lens references to the Kaggle: 

[https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)

[https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset)

The columns and this merging will be explored in the preliminary report. 

## Algorithms / Techniques 
**Aproiri (Ethan):** The dataset can be split into groups – for example, all the 1 rating a person gives creates a set, and then find associations using Aproiri. This will be able to predict if a user gives two things the same rating that they will give another movie the same rating. 

Essentially, apriori functions by counting the frequency of the item for each user in their set, pruning the infrequent ones, counting the support and eliminating the infrequent ones over and over until the subsets that lead to another remain. 

**K Nearest Neighbours (William):* We can select a profile or movie and select similar profiles or movies to them so that their movies can be looked at. 

Essentially, this would function by measuring the Euclidean distance between profiles or movies based on numerical values or sets (i.e. genres can contribute).  

Density-based clustering (Nick): Profiles and movies can be grouped into clusters by density so that people can be put into groups based on their preferences. 

This essentially functions by highlighting areas with high density as a group until it reaches a point where it is not dense. The idea behind this is that users would be separated into their own groups of interests with borders in density. 

## Measurements to evaluate 

We can take a movie out of a person’s profile and see if the movie recommendation system recommends that same movie with the same or worse score. Furthermore, as there are many rows then n-fold cross-validation can also be used from this.
## Preliminary 

Visible in preliminary.ipnb or the html format for easy opening. 
