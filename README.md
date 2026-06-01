# Movie Recommendation System (Cornac + CTR)
A script that trains a hybrid recommender on the MovieLens 100k dataset using Cornac.
It combines collaborative filtering signals (user-item ratings) with content signals (movie plot text) via the Collaborative Topic Regression (CTR) model, evaluates it, and prints sample recommendations for a few users

# What is does
1. Loads MovieLens 100K rating feedback
2. Loads movie plot summaries and wraps them in a Cornac TextModality (content-based part of hybrid model)
3. Downloads u.item from GroupLens to build a movieId -> title lookup so output shows readable movie names instead of raw IDS.
4. Splits data 80/20 (train/test) with RatioSplit, treating ratings >= 4.0 as positive relevance
5. Trains CTR model
6. Prints each sample user's known liked movies (with ratings the users gave) alongside model's top recommendations (with predicted scores), with already-seen movies excluded from recommendations

# Ratings vs predicted scores
- Known positives show the user's actual ratings, taken from training matrix
- Recommendations show CTR's predicted scores. 
