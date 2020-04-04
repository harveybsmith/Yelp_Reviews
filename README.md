# Yelp_Reviews
Natural Language Processing of Yelp Reviews to predict ratings

In this NLP project we will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews.

We will use the Yelp Review Data Set from Kaggle.

Each observation in this dataset is a review of a particular business by a particular user.

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.

The "cool" column is the number of "cool" votes this review received from other Yelp users.

All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.

The "useful" and "funny" columns are similar to the "cool" column.

## Feature Engineering 
I created a column called "Text Length" which created numeric values representing a messages length

## EDA

![](/text_length_fg.png)

This facetgrid shows histograms for each "star" ordinal category.  The y axis representing the counts and the x axis representing the text lengths, organized into 8 bins.  As one can notice, reviews with higher ratings have more reviews but the majority of the reivew lengths are short.

![](/starts_countplot.png)

Next a simple countplot shows that the majority of the Yelp listings in our dataset have high ratings of 4 or 5 stars

Doing an aggregation groupby on our dataset by 'stars' gives us some interesting observations:

	cool	         useful		funny	        text legnth
`stars				
`1	`0.576769	1.604806	1.056075	826.515354
`2	`0.719525	1.563107	0.875944	842.256742
`3	`0.788501	1.306639	0.694730	758.498289
`4	`0.954623	1.395916	0.670448	712.923142
`5	`0.944261	1.381780	0.608631	624.999101
