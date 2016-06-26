
import pandas as pd 
import numpy as np
import csv as csv 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Eliminate false positive SettingWithCopyWarning
pd.options.mode.chained_assignment = None

##Load Data from CSV
train_df = pd.read_csv("data/train.csv", header = 0) #training data
test_df = pd.read_csv("data/test.csv", header = 0) #test data

test_ids=test_df['PassengerId'].values



def clean_data(passed_df):
	
	##Convert "female" and "male" to be 0 and 1
	passed_df['Gender']=passed_df['Sex'].map({"female":0, "male":1}).astype(int)

	##convert empty embarkment to most common embarkment value
	if len(passed_df.Embarked[passed_df.Embarked.isnull() ] ) > 0:
		passed_df.Embarked[passed_df.Embarked.isnull() ] = passed_df.Embarked.dropna().mode().values

	##Convert embarked data into number values with a dictionary	
	Ports = list(enumerate(np.unique(passed_df['Embarked'])))
	Ports_dict = { name: i for i, name in Ports}
	passed_df.Embarked = passed_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

	#replace missing ages with median age
	median_age=passed_df.Age.dropna().median()
	if len(passed_df.Age[passed_df.Age.isnull() ]) > 0:
		median_age=np.zeros(3)
		for f in range(0,3):
			median_age[f]=passed_df[(passed_df.Pclass == f+1)]['Age'].dropna().median()
		for f in range(0,3):
			passed_df.loc[((passed_df.Age.isnull()) & (passed_df.Pclass == f+1)), 'Age']=median_age[f]
			#passed_df.loc[ (passed_df.Age.isnull()), 'Age' ] = median_age

	#replace missing fares with median fare for that class
	if len(passed_df.Fare[passed_df.Fare.isnull() ]) > 0:
		median_fare=np.zeros(3)
		for f in range(0,3):
			median_fare[f]=passed_df[(passed_df.Pclass == f+1)]['Fare'].dropna().median()
		for f in range(0,3):
			passed_df.loc[((passed_df.Fare.isnull()) & (passed_df.Pclass == f+1)), 'Fare']=median_fare[f]

	#drop fields not used for machine learning
	passed_df = passed_df.drop(["Name", "Sex", "Ticket", "Cabin", "PassengerId"], axis=1)
	
	return passed_df

##Clean up data
train_df=clean_data(train_df)
test_df=clean_data(test_df)

#convert data to a list for the forest algorithm
train_data = train_df.values
test_data = test_df.values

print "Training..."
forest=RandomForestClassifier(n_estimators=10000)
forest=forest.fit(train_data[0::,1::], train_data[0::,0])

print "Predicting..."
output=forest.predict(test_data).astype(int)

predictions_file = open("titanicpredictions.csv", 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids, output))
predictions_file.close()
print "Done."

#print train_data[0::,1::]
#print train_data[0::,0]
#print test_df.head()