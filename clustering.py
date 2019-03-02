import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import pickle 


#loading the dataset
mat = pd.read_csv("student-mat.csv", sep=';')

"""mat_copy_y = mat_copy['G3']
mat_copy = mat_copy.drop(columns=['G3']) """
#preprocessing of the data
def preprocess_features(X):
   
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  

        # Collect the revised columns
        output = output.join(col_data)

    return output
	
mat = preprocess_features(mat)

#spliting dataset into input and output for the model

"""kmeans = KMeans(n_clusters=3, random_state=0).fit(mat_copy)
y_kmeans = kmeans.predict(mat_copy)"""
y = mat['G3']
X = mat.drop(columns=['G3'])
#print(X)

#feature selection 
mat_reformed = preprocess_features(mat)

#k-means for labeling the dataset
kmeans = KMeans(n_clusters=4, random_state=0).fit(mat_reformed)
y_kmeans = kmeans.predict(mat_reformed)
#print(y_kmeans)

mat['cluster_index'] = 'na'
for index,row in mat.iterrows():
	mat.at[index, 'cluster_index'] = y_kmeans[index]

for index,row in mat.iterrows():
	if row["cluster_index"]==0:
		mat.at[index, 'cluster_index'] = 'A'
	if row["cluster_index"]==1:
		mat.at[index, 'cluster_index'] = 'B'
	if row["cluster_index"]==2:
		mat.at[index, 'cluster_index'] = 'C'
	if row["cluster_index"]==3:
		mat.at[index, 'cluster_index'] = 'D'
		
y = mat['cluster_index']
X = mat.drop(columns=['cluster_index'])

print(X)
print(y)
print(type(X))
print(type(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=4)
#print(y_test)
neigh.fit(X_train, y_train) 
result = neigh.predict(X_test)

acc = accuracy_score(y_test, result)
print("Accuracy of KNN algorithm is : ",acc,"\n")

#feature extraction or variable selection

bestfeatures = SelectKBest(score_func=chi2, k=15)
fit = bestfeatures.fit(X,y)
print(type(X))
print(type(y))
mat_copyscores = pd.DataFrame(fit.scores_)
mat_copycolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([mat_copycolumns,mat_copyscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
top_5 = featureScores.nlargest(5,'Score')
top_15 = featureScores.nlargest(20,'Score')

print("The top rated features are: ")
print(top_15,"\n")

#storing of variable most effective towards clustering
headers = []
score = []
for index,row in top_5.iterrows():
	#print(type(row["Specs"]))
	headers.append(row["Specs"])
	#score.append
X_refine = X.copy()
X_refine = X_refine[headers]

kmeans_refined = KMeans(n_clusters=4, random_state=0).fit(X_refine)
y_kmeans_refined = kmeans_refined.predict(X_refine)

mat['cluster_index_refine'] = 'na'
for index,row in mat.iterrows():
	mat.at[index, 'cluster_index_refine'] = y_kmeans_refined[index]
	
#print(mat)

#again evaluating the model
grades_refined = []
for index, row in mat.iterrows():
	details_refined = []
	
	if row["cluster_index"] not in grades_refined:
		categories_refined = {}
		
		categories_refined[row["cluster_index"]] = details_refined
		details_refined.append(row["cluster_index_refine"])
		categories_refined[row["cluster_index"]] = details_refined

		pass
	else:
		details_refined.append(row["cluster_index_refine"])
		categories_refined[row["cluster_index"]] = details_refined
		pass
		
	grades_refined.append(categories_refined)
	
#print(grades_refined)

#list for labeling the cluster.
A= []
B =[]
C =[]
D =[]

#storing the lables.
for grade in grades_refined:
	for key,value in grade.items():
		if key == 'A':
			for val in value:
				A.append(val)
		if key == 'B':
			for val in value:
				B.append(val)
		if key == 'C':
			for val in value:
				C.append(val)
		if key == 'D':
			for val in value:
				D.append(val)

#get the maximum of a label
Am,An = Counter(A).most_common(1)[0]
Bm,Bn = Counter(B).most_common(1)[0] 
Cm,XC = Counter(C).most_common(1)[0] 
Dm,Dn = Counter(D).most_common(1)[0] 
			
def labelling(x):
	if Am == x:
		return 'A'
	if Bm == x:
		return 'B'
	if Cm == x:
		return 'C'
	if Dm == x:
		return 'D'
	if Am == x:
		return 'A'
	if Bm == x:
		return 'B'
	if Cm == x:
		return 'C'
	if Dm == x:
		return 'D'
	if Am == x:
		return 'A'
	if Bm == x:
		return 'B'
	if Cm == x:
		return 'C'
	if Dm == x:
		return 'D'
	if Am == x:
		return 'A'
	if Bm == x:
		return 'B'
	if Cm == x:
		return 'C'
	if Dm == x:
		return 'D'
	
label_0_most_common = labelling(0)
label_1_most_common = labelling(1)
label_2_most_common = labelling(2)
label_3_most_common = labelling(3)

#labeling the cluster of refine kmeans
for index,row in mat.iterrows():
	if row["cluster_index_refine"]==0:
		mat.at[index, 'cluster_index_refine'] = label_0_most_common
	if row["cluster_index_refine"]==1:
		mat.at[index, 'cluster_index_refine'] = label_1_most_common
	if row["cluster_index_refine"]==2:
		mat.at[index, 'cluster_index_refine'] = label_2_most_common
	if row["cluster_index_refine"]==3:
		mat.at[index, 'cluster_index_refine'] = label_3_most_common
#print(mat)


y_refine = mat['cluster_index_refine']
X_refine = mat.drop(columns=['cluster_index_refine','cluster_index'])

X_train, X_test, y_train, y_test = train_test_split(X_refine, y_refine, test_size=0.15, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, y_train) 
result = neigh.predict(X_test)
print(X_test)
print(result)
acc = accuracy_score(y_test, result)
print("The accuracy after Feature selection is : ",acc)
print("features : ")

failure = int(input("enter if you have ever failed : "))
absence = int(input("enter the number of days you likely to absent : "))
G1 = int(input("enter the G1 score : "))
G2 = int(input("enter the G2 score : "))
G3 = int(input("enter the G3 score : "))

feed = []
feed.append(failure)
feed.append(absence)
feed.append(G1)
feed.append(G2)
feed.append(G3)

new_header = list(X_test.columns.values)
my_df  = pd.DataFrame(columns = new_header)

for index,row in my_df.iterrows():
	my_df.at[index, row] = '0'

my_df.at['C', 'failures'] = failure
my_df.at['C', 'G3'] = G3
my_df.at['C', 'G1'] = G1
my_df.at['C', 'G2'] = G2
my_df.at['C', 'absences'] = absence

my_df = my_df.fillna(0)
result = neigh.predict(my_df)

def perfomance(result):

	if result=='A':
		return 'Very Good'
	if result=='B':
		return 'Good'
	if result=='C':
		return 'Average'
	if result=='D':
		return'Bad'
		
final_result = perfomance(result)
print("Your perfomance is predicted as ",final_result)

neighFile = open('neigh.pckl', 'wb')
pickle.dump(neigh, neighFile)
neighFile.close()