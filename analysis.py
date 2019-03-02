import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns

mat = pd.read_csv("student-mat.csv", sep=';')

print(mat)
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
headers = list(mat.columns.values)
#print(headers)
#spliting dataset into input and output for the model

kmeans = KMeans(n_clusters=3, random_state=0).fit(mat)
y_kmeans = kmeans.predict(mat)
#print(y_kmeans)
mat['final_grade'] = 'na'
#print(y_kmeans)
for index,row in mat.iterrows():
	if y_kmeans[index]==1:
		mat.at[index, 'final_grade'] = 'very good'
	elif y_kmeans[index]==2:
		mat.at[index, 'final_grade'] = 'good'
	else:
		mat.at[index, 'final_grade'] = 'bad'

print(mat)
#plot the graph of number of students with different grades.		
plt.figure(figsize=(8,6))
sns.countplot(mat.final_grade, order=['bad','good','very good'], palette='Set1')
plt.title('Final Grade - Number of Students',fontsize=20)
plt.xlabel('Final Grade', fontsize=16)
plt.ylabel('Number of Student', fontsize=16)
plt.show()

#to show the impact of parent's education.
good = mat.loc[mat.final_grade == 'good']
very_good = mat.loc[mat.final_grade == 'very good']
bad = mat.loc[mat.final_grade == 'bad']
good_very_good = pd.concat([good,very_good])

father_education = 0
mother_education = 0

for index,row in good_very_good.iterrows():
	if row["Medu"]>1:
		mother_education += 4
	if row["Fedu"]>1:
		father_education += 4

objects = ("Mother's Education","Father's Eduaction")
y_pos = np.arange(len(objects))
performance = [mother_education,father_education]
 
plt.bar(y_pos, performance, align='center', alpha=1)
plt.xticks(y_pos, objects)
plt.title("No.of Educated parents of students having good perfomance")
plt.show()

#Effect of going out on students having bad perfomance
go_out_1 = 0
go_out_2 = 0
go_out_3 = 0
go_out_4 = 0
go_out_5 = 0

go_out = mat.loc[mat.final_grade == 'goout']
for index,row in bad.iterrows():
	if row["goout"]==1:
		go_out_1 +=1
	if row["goout"]==2:
		go_out_2 +=1
	if row["goout"]==3:
		go_out_3 +=1
	if row["goout"]==4:
		go_out_4 +=1
	if row["goout"]==5:
		go_out_5 +=1
		
"""object = ('go_out_1','go_out_2','go_out_3','go_out_4','go_out_5')
y_po = np.arange(len(object))
performanc = [go_out_1,go_out_2,go_out_3,go_out_4,go_out_5]
 
plt.barh(y_po, performanc, align='center', alpha=1)
plt.yticks(y_po, object)
plt.title('No. of Students')
 
plt.show()"""
 
 
#for good Students effect of going out.
go_out_1_good = 0
go_out_2_good = 0
go_out_3_good= 0
go_out_4_good = 0
go_out_5_good = 0

go_out = mat.loc[mat.final_grade == 'goout']
for index,row in good.iterrows():
	if row["goout"]==1:
		go_out_1_good +=1
	if row["goout"]==2:
		go_out_2_good +=1
	if row["goout"]==3:
		go_out_3_good +=1
	if row["goout"]==4:
		go_out_4_good +=1
	if row["goout"]==5:
		go_out_5_good +=1
		
#effect of going out on students having good and bad perfomance
objectg = ('go_out_1_bad','go_out_2_bad','go_out_3_bad','go_out_4_bad','go_out_5_bad','go_out_1_good','go_out_2_good','go_out_3_good','go_out_4_good','go_out_5_good')
y_pog = np.arange(len(objectg))
performancg = [go_out_1,go_out_2,go_out_3,go_out_4,go_out_5,go_out_1_good,go_out_2_good,go_out_3_good,go_out_4_good,go_out_5_good]
 
plt.barh(y_pog, performancg, align='center', alpha=1)
plt.yticks(y_pog, objectg)
plt.title('No. of Students going out')
 
plt.show()


