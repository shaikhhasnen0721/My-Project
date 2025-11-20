# Liberaries
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

#Dataset to DataFrame convert
df1 = pd.read_csv("house_price_data.csv")
print(df1.head())
print(df1.shape)

print(df1.groupby('area_type')['area_type'].agg('count'))

#DATA CLEANING

# Drop the columns (area_type, availability)
df2 = df1.drop(['area_type', 'availability'],axis='columns')
print(df2.shape)

# # Check the null value in dataset
print(df2.isnull().sum())

# Drop the null rows
df3 = df2.dropna()
print(df3.isnull().sum())

#Create new column 'bhk' in size column
df3 = df3.copy()
df3.loc[:, 'bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.bhk.unique())
#
# we can find this type values in dataset 3067 - 8156
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
print(df3[~df3['total_sqft'].apply(is_float)].head(10))

# we can find this type values in dataset 3067 - 8156 and convert this type 1270.0 with the help of
# this function
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
print(df4.head(5))

#We can add new column of 'price_per_sqft'
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*1/df5['total_sqft']
print(df5.head())

#Save upgraded CSV file
df5.to_csv("hpp_excet.csv",index=False)

#Print all the location in ascending type
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
print(location_stats)

# find len of locations who is >10
print(len(location_stats[location_stats>10]))
print(len(location_stats))
print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

print(len(df5.location.unique()))

#Find the other name location in dataset using this function
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

#Print dataset first 10 rows
print(df5.head(10))

#Find the total_sqft who is <300
print(df5[df5.total_sqft/df5.bhk<300].head())

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
print(df7.shape)

#Print histogram
# matplotlib.rcParams["figure.figsize"] = (20,10)
# plt.hist(df7.price_per_sqft,rwidth=0.8)
# plt.xlabel("Price Per Square Feet")
# plt.ylabel("Count")
# plt.show()

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
df8 = df7.copy()
print(df8.shape)

#Drop the column
df9 = df8.drop(['size', 'price_per_sqft', 'category'], axis = 'columns')
print(df9.head(3))

#Make dummy data using dummy method
dummies = pd.get_dummies(df9.location)
print(dummies.head(3))

df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
print(df10.head())

df11 = df10.drop('location',axis='columns')
print(df11.head(2))
print(df11.shape)

#Drop the price column
X = df11.drop(['price'],axis='columns')
print(X.head(3))
print(X.shape)

y = df11.price
print(y.head(3))
print(len(y))

#CREATE MODULE USING LINEAR REGRASSION
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
print(lr_clf.score(X_test,y_test))

# Find the ShuffleSplit and Crosss val score of module
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(cross_val_score(LinearRegression(), X, y, cv=cv))

print(X.columns)

def predict_price(location, sqft, bhk, area_type):
    """
    area_type must be: 'Urban' or 'Rural'
    """

    # Create input row
    x = np.zeros(len(X.columns))

    # numeric features
    x[0] = sqft
    x[1] = bhk

    # ---- Location One-Hot Handling ----
    if location in X.columns:
        loc_index = np.where(X.columns == location)[0][0]
        x[loc_index] = 1

    # ---- Urban / Rural One-Hot Handling ----
    # Check if dataset contains one-hot encoded area columns
    area_col = None

    # Case 1: column names like 'urban' or 'rural'
    if area_type.lower() in X.columns:
        area_col = area_type.lower()

    # Case 2: one-hot encoded names like 'area_Urban', 'area_Rural'
    encoded_name = f"area_{area_type}"
    if encoded_name in X.columns:
        area_col = encoded_name

    # Set the value if column exists
    if area_col is not None:
        area_index = np.where(X.columns == area_col)[0][0]
        x[area_index] = 1

    # Convert to DataFrame to keep feature names
    x_df = pd.DataFrame([x], columns=X.columns)

    return float(lr_clf.predict(x_df)[0])

print(predict_price("7th Phase JP Nagar", 1080, 2, "Urban"))
print(predict_price('Chandapura',800, 2, 'Rural'))

#Make pikle file
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

#Make json file
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))