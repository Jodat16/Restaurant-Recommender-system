# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import io

df1=pd.read_csv("geoplaces2.csv")
df2=pd.read_csv("rating_final.csv")
ratings=pd.merge(df1,df2) # merge two df's
ratings=ratings[['placeID','name','userID','rating']] # take needed columns
ratings['userID'] = ratings['userID'].str[1:]
ratings.dropna(inplace=True)
ratings.head() # show new dataframe

"""# Data Analysis"""

ratings.info() # infos about samples, features and datatypes
print('Shape of Data :')
ratings.shape

ratings.name.nunique() # unique number of restaurants

"""# Data Visualization"""

sns.countplot(x=ratings.rating); # plot the ratings

# Group the data by item ID and calculate the number of ratings and average rating for each item
item_ratings = ratings.groupby('placeID').agg({'rating': ['count', 'mean']}).reset_index()
# Rename the columns
item_ratings.columns = ['placeID', 'num_ratings', 'avg_rating']
# Create a pivot table showing the correlation between the number of ratings and average rating for each book
pivot = pd.pivot_table(item_ratings, index='num_ratings', columns='avg_rating', values='placeID')
# Create a heatmap of the pivot table
plt.figure(figsize=(20,20))
sns.heatmap(pivot, cmap='RdBu_r');

"""# Popular Restaurants
### using popularity based recommender system (not based on ratings)
"""

# function to calculate popularity stats
def popularity_based_rec(df, group_col, rating_col):
    # group by title and get size, sum and mean values
    grouped = df.groupby(group_col).agg({rating_col: [np.size, np.sum, np.mean]})
    # most popular mean value on top
    popular = grouped.sort_values((rating_col, "sum"), ascending=False)
    return popular

# call function and show top 5 restaurants
popularity_stats = popularity_based_rec(ratings, "name", "rating")
popularity_stats.head(10) # show top 5 restaurants

"""# K-Nearest Neighbor based recommender system"""

#sort the restaurants from largest to smallest according to its mean ratings
itemProperties = ratings.groupby("placeID").agg({"rating": [np.size, np.mean]})
itemProperties.head()

# calculate their percentages
itemNumRatings = pd.DataFrame(itemProperties["rating"]["size"])
itemNormalizedNumRatings = itemNumRatings.apply(lambda x: (x-np.min(x)) / (np.max(x) - np.min(x)))
itemNormalizedNumRatings.head() # show last 5 entries

ratings.to_csv("ratings.csv")

df=pd.read_csv('ratings.csv')
df.head()

# store all restaurants in a dictionary with their id's, names ratings, number of ratings and average ratings
itemDict = {} # create an empty item Dictionary
# Read in the ratings data from the CSV file
with open('ratings.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)  # skip the first row
    for row in csv_reader:
        if row[1] == '' or row[2] == '' or row[3] == '' or row[4] == '': #skip empty rows
            continue
        # get the right columns
        itemID = int(row[1])
        name = row[2]
        userID = int(row[3])
        rating = int(row[4])

        if itemID not in itemDict:
            itemDict[itemID] = {'name': name, 'ratings': [], 'numRatings': 0, 'totalRating': 0}
        itemDict[itemID]['ratings'].append(rating)
        itemDict[itemID]['numRatings'] += 1
        itemDict[itemID]['totalRating'] += rating

# Calculate the average rating for each item
for itemID in itemDict:
    item = itemDict[itemID]
    name = item['name']
    ratings = item['ratings']
    numRatings = item['numRatings']
    totalRating = item['totalRating']
    avgRating = totalRating / numRatings
    itemDict[itemID] = {'name': name, 'ratings': ratings, 'numRatings': numRatings, 'avgRating': avgRating}

itemDict

# function that finds the distance of an item from another item - SIMILARITY
def ComputeDistance(a, b):
    # Find the common ratings(by common user) for both item
    common_ratings = [rating for rating in a['ratings'] if rating in b['ratings']]

    # If there are no common ratings, the distance is infinity
    if len(common_ratings) == 0:
        return float('inf')

    # If the lists of ratings are not the same length, return infinity
    if len(a['ratings']) != len(b['ratings']):
        return float('inf')

    # Calculate the sum of the squared differences between the ratings
    sum_squared_differences = sum([(a['ratings'][i] - b['ratings'][i]) ** 2 for i in range(len(common_ratings))])

    # Return the square root of the sum of squared differences, which is the distance between the two items
    return sum_squared_differences ** 0.5

# function to get K-Nearest Neighbors
def getNeighbors(itemID, K):
    # Get the item object for the given item ID
    target_item = itemDict[itemID]

    # Create a list of tuples (distance, itemID) for all items in the dictionary
    distances = [(ComputeDistance(target_item, itemDict[itemID]), itemID) for itemID in itemDict if itemDict[itemID]['name'] != target_item['name']]

    distances.sort()

    return distances[:K]

# get the smallest distances as a list
neighbors = getNeighbors(134999, 30)
# Print the item names and distances of the nearest neighbors
for distance, itemID in neighbors:
    print(f"{itemDict[itemID]['name']}: {distance:.2f}")

"""# User based Recommender System"""

ratings=pd.read_csv("ratings.csv")

# Matrix Factorization (pivot_table)
userratings=ratings.pivot_table(index=['name'],columns=["userID"],values="rating")
userratings.tail(10)

# an example of the correaltion between '1001' and '1104'
userratings[[1001,1104]].corr()

#show user with most correlation with user 1001
user = userratings[1001]
corr_users = userratings.corrwith(user).sort_values(ascending=False).to_frame('corr').dropna()
corr_users

"""##Prediction Function"""

def drop_rest_and_users(data_table, user_ID, restaurant_name):
  #drop all rows/restaurants which are not rated by user_ID (lets say 1001)
  df_filtered = data_table.dropna(subset=[user_ID])

  #drop columns/users that haven't rated curr_restaurant
  df_filtered = df_filtered.drop(columns=userratings.columns[userratings.loc[restaurant_name].isnull()])
  return df_filtered

#Take input
curr_user = int(input('Enter user for whom you want prediction : ')) #1001
restaur = input('Enter name of the restaurant for which you want prediction  : ') #Cafe Chaires

#Just to show table with removed users who have not rated current restaurant and restaurants which are not rated by current user
filtered_df = drop_rest_and_users(userratings, curr_user, restaur)
filtered_df

#Prediction function
def predict(user_pred, correlated):

  common_indexes = list(user_pred.index.intersection(correlated.index))
  top_indexes = user_pred.loc[common_indexes].nlargest(2).index.tolist()

  if top_indexes==0:
    print('PREDICTION NOT POSSIBLE DUE TO LACK OF DATA')
    return 0

  Rating = 0
  Rating_numerator = 0
  Rating_denominator = 0
  #print(top_indexes)

  for similar_user in top_indexes:
    Rating_numerator = Rating_numerator + (user_pred.loc[similar_user]*correlated.loc[similar_user])
    Rating_denominator = Rating_denominator + abs(correlated.loc[similar_user])

  Rating = Rating_numerator/Rating_denominator
  return Rating

#find correlation and PREDICT
user = userratings[curr_user]
corr_users = userratings.corrwith(user).sort_values(ascending=False).to_frame('corr').dropna()

#get id's of correlated users
id_corr_users = corr_users.index.tolist()

#get correlated users who have rated current restaurant
have_rated = userratings.loc[restaur,id_corr_users].dropna()

#have_rated.loc[1055]
rating = predict(have_rated, corr_users)
ratings = int(rating)
print(f"Predicted rating for {restaur} by user {curr_user} is : {ratings} \n\n\n")