import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

matplotlib.rcParams["figure.figsize"] = (20, 10)

df1=pd.read_csv("house_price.csv")

#remove unnecessary features
df2 = df1.drop(['Postcode', 'Bedroom2', 'BuildingArea'], axis='columns')

#identify object datatype(numeric,boolean,char,date but not string)
#print(df2.select_dtypes(['object']).columns)

# Convert objects to categorical variables
obj_cats = ['Suburb', 'Method', 'SellerG', 'Regionname', 'CouncilArea']

for colname in obj_cats:
    df2[colname] = df2[colname].astype('category')

df2['Date'] = pd.to_datetime(df2['Date'])

#convert rows to column and column to rows(temporary)
#print(df2.describe().transpose())


#print(df2.isnull().sum()/len(df2)*100)
#significant pc of buildingarea feature is missing so we will drop this column alltogeher in line 12
#same for year built

df2=df2.dropna()

#outlier removal

#print(df2.select_dtypes(['category']).columns)

# Abbreviate Regionname categories
df2['Regionname'] = df2['Regionname'].map({'Northern Metropolitan': 'N Metro',
                                            'Western Metropolitan': 'W Metro',
                                            'Southern Metropolitan': 'S Metro',
                                            'Eastern Metropolitan': 'E Metro',
                                            'South-Eastern Metropolitan': 'SE Metro',
                                            'Northern Victoria': 'N Vic',
                                            'Eastern Victoria': 'E Vic',
                                            'Western Victoria': 'W Vic'})


#boxplot for features containing less variables
#graph for catagorical features using boxplot feature of seaborn
#the line in the middle of the box in boxplot represents median

# Suplots of categorical features v price
sns.set_style('darkgrid')
f, axes = plt.subplots(2, 2, figsize=(15, 15))

# Plot [0,0]
sns.boxplot(data=df2, x='Type', y='Price', ax=axes[0, 0])
axes[0,0].set_xlabel('Type')
axes[0,0].set_ylabel('Price')
axes[0,0].set_title('Type v Price')

# Plot [0,1]
sns.boxplot(x='Method', y='Price', data=df2, ax=axes[0, 1])
axes[0, 1].set_xlabel('Method')
axes[0,1].set_ylabel('Price')
axes[0,1].set_title('Method v Price')

# Plot [1,0]
sns.boxplot(x = 'Regionname', y = 'Price', data = df2, ax = axes[1,0])
axes[1,0].set_xlabel('Regionname')
axes[1,0].set_ylabel('Price')
axes[1,0].set_title('Region Name v Price')

# Plot [1,1]
sns.boxplot(x = 'CouncilArea', y='Price', data = df2, ax=axes[1,1])
axes[1, 1].set_xlabel('CouncilArea')
axes[1, 1].set_ylabel('Price')
axes[1, 1].set_title('CouncilArea v Price')

plt.show()

#scatter plot for features containing large number of variables
sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (20,30))

# Plot [0,0]
axes[0,0].scatter(x = 'Suburb', y = 'Price', data=df2, edgecolor = 'b')
axes[0,0].set_xlabel('Suburb')
axes[0,0].set_ylabel('Price')
axes[0,0].set_title('Suburb v Price')

# Plot [0,1]
axes[0,1].scatter(x = 'Distance', y='Price', data=df2, edgecolor = 'b')
axes[0,1].set_xlabel('Distance')
# axes[0,1].set_ylabel('Price')
axes[0,1].set_title('Distance v Price')

# Plot [1,0]
axes[1,0].scatter(x='Bathroom', y='Price', data = df2, edgecolor = 'b')
axes[1,0].set_xlabel('Bathroom')
axes[1,0].set_ylabel('Price')
axes[1,0].set_title('Bathroom v Price')

# Plot [1,1]
axes[1,1].scatter(x = 'CouncilArea', y='Price', data=df2, edgecolor='b')
axes[1,0].set_xlabel('CouncilArea')
axes[1,1].set_ylabel('Price')
axes[1,1].set_title('CouncilArea v Price')


plt.show()

#remove outlier

df3 = np.sort(df2['Price'])

Q1 = np.percentile(df3, 25, interpolation='midpoint')
Q3 = np.percentile(df3, 75, interpolation='midpoint')
IQR = Q3-Q1

print(IQR)

outlier =[]
for x in df3:
    if ((x> Q3 + 1.5 * IQR ) or (x<Q1 - 1.5 * IQR )):
         outlier.append(x)
print(' outlier length  is', len(outlier))



#this part needs correction
sample = pd.DataFrame(outlier, columns=['Price'])
print(sample)


df2=df2[df2.Price<outlier[0]]  #Drop all rows where price<outlier's minimum value

print(df2.shape)

#plotting correlation graph
plt.figure(figsize=(10, 6))
sns.heatmap(df2.corr(), cmap='coolwarm', linewidth=1, annot=True, annot_kws={"size": 9})
plt.title('Variable Correlation')
plt.show()

print(df2.select_dtypes(['float64', 'int64']).columns)

dummy = pd.get_dummies(df2.Suburb) #make suburb column feature instead of row for regression
df4 = pd.concat([df2, dummy], axis=1)
df5=df4.drop(['Suburb'], axis='columns')
print(df5.head(10))
##############################################################################################################################
X=df5.drop(['Address', 'Price', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname'], axis='columns')
print(X.head(10))
Y=df5.Price

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.20, random_state=0)



clf=tree.DecisionTreeRegressor()
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

print(cross_val_score(tree.DecisionTreeRegressor(), X, Y, cv=cv))#accuracy is around 65% on avg


y_pred = clf.predict(X_test)
###############################################################################
def predict_price(Suburb,Rooms,Distance,Bathroom,Car):#function to predict price
    loc_index = np.where(X.columns==Suburb)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = Rooms
    x[1] = Distance
    x[2] = Bathroom
    x[3] = Car
    if loc_index >= 0:
        x[loc_index] = 1

    return clf.predict([x])[0]

print(predict_price('Abbotsford',2,2.5,1,1))