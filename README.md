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

# Correlations
Doing an aggregation mean() on a groupby of our dataset by 'stars' gives us some interesting observations:

1.  The reviews marked as 'cool', while not frequently used, have a strong correlation with more stars
2.  Reviews marked as 'useful' by others, tend to correlate with lower stars
3.  Reviews marked as funny strongly correlates with with lower stars. The funnier, the poorer the rating
4.  Longer tenxt length correlated with lower rating.

![](/groupby.png)

Observiing the correlations between the individual features to each other using a heatmap reveals that:
1. 'Cool' is hardly correlated at all with the other features of a review
2.  'Useful' is strongly correlated with 'Funny'
3.  'Funny' is strongly correlated with 'Text Length'
4.  'Text Length is correlated with 'Useful' by not strongly (.7)

![](/corr_heatmap.png)

## NLP Classification Task

To simplify things I only looked at ratings with a 1 star or 5 stars.  So our model will tell us if a listing has with a very poor review or a very good review.  In that was it is more binary

* First we vectorizer our text data with CountVectorizer, where each each listing is a row, and every unique word in all the reivews is transformed into a column, and each value is either a 1 if the word appears in the listing, or 0 if not. This create a very sparse matrix. We call this a 'Bag of Words'.  
 * Then we perform a Multinomial Naive Bayes on this matrix.  Multinomial is better at dealing with sparse data.  It doesnt need to have each word as dependent as each other word as is "Naively" assumes a correlation between them
 
               precision    recall  f1-score   support

           1       0.88      0.70      0.78       228
           5       0.93      0.98      0.96       998

    accuracy                           0.93      1226
   macro avg       0.91      0.84      0.87      1226
weighted avg       0.92      0.93      0.92      1226

The results are good.  0.92 accuracy.  

# Using TDIDF

Then I try using TDIDF, Term Frequency Inverse Document Frequecy.  This makes the sparse matrix of bag of words less sparse and gives a continuous value for each word from 0 to 1.  Each word is now taken as the proportion of words in the document (review), the Term Frequency.  The second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.

Example:

Consider a document containing 100 words wherein the word cat appears 3 times.

The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.

We create a simple Pipeline using 

`pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tdidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])`

When we fit the pipeline and then use predict to create predictions for our test data we get the folowing results

              precision    recall  f1-score   support

           1       0.00      0.00      0.00         0
           5       1.00      0.82      0.90       818

    accuracy                           0.82       818
`  macro avg       0.50      0.41      0.45       818
weighted avg       1.00      0.82      0.90       818

