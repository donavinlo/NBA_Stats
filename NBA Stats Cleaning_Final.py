#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# In[12]:


df1 = pd.read_csv('NBA Stats.csv')


# In[13]:


del df1['Unnamed: 0']


# In[14]:


#Need to Convert Date to Date/Time
#Parse the date and create seperate column for months
df1['Date']= pd.to_datetime(df1['Date'], format = '%m/%d/%Y', errors = 'coerce')
df1['Year']= df1['Date'].dt.year
df1['Month'] = df1['Date'].dt.month
df1['Day_num'] = df1['Date'].dt.day



# In[15]:


del df1['Date']


# In[16]:


df1.head()


# In[17]:


df1.columns


# In[18]:


#Convert Game, Home, Win/Loss,Year, Day_num to Category
df1['Game'] = pd.Categorical(df1.Game, ordered = True)
df1['Home'] = pd.Categorical(df1.Home)
df1['WINorLOSS'] = df1['WINorLOSS'].map({'W':1, 'L':0})
df1['Year']=pd.Categorical(df1.Year)
df1['Day_num']=pd.Categorical(df1.Day_num)
df1['Team']= pd.Categorical(df1.Team)
df1['Opponent'] = pd.Categorical(df1.Opponent)
#To Practice replacing values, change #'s in month to their corresponding Month abbreviation'
df1['Month'] = df1['Month'].map({10: 'Oct', 11: 'Nov', 12:'Dec', 1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr'})


# In[19]:


df1['Month']= pd.Categorical(df1.Month)


# In[20]:


#Make a for loop so all shooting percentage columns are sent from decimals to whole numbers
#Shooting percentage columnns are indicated by the period at the end of the column header
#First Need to Rename Columns
df1.rename(columns = {'Home':'Location','FieldGoals':'FG', 'FieldGoalsAttempted':'FGA','FieldGoals.':'FG%', 'X3PointShots':'3ptShots', 'X3PointShotsAttempted':'3ptShotsAtt',
                      'X3PointShots.': '3ptShot%', 'FreeThrows':'FT', 'FreeThrowsAttempted':'FTA', 'FreeThrows.':'FT%', 'Assists':'AST', 
                      'Steals':'STL','OffRebounds':'OREB', 'TotalRebounds':'REB', 'Blocks':'BLK', 'Turnovers':'TO'}, inplace = True)


# In[21]:


#Convert shooting percentages to whole numbers
shot_perc = ['FG%', '3ptShot%', 'FT%']
for i in shot_perc:
    df1[i] = df1[i] *100


# In[22]:


#Probably won't need the day number of the month. Let's get rid of this column
del df1['Day_num']
df1['Point_Dif'] = df1['TeamPoints'] - df1['OpponentPoints']


# In[23]:


#rearrange a dataframe so into the order we want the columns
df1= df1[['Team', 'Game', 'Year', 'Month', 'Location', 'Opponent', 'WINorLOSS', 'TeamPoints', 'Point_Dif', 'FG', 'FGA', 'FG%', '3ptShots', '3ptShotsAtt', '3ptShot%', 'FT', 'FTA', 'FT%', 'OREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'TotalFouls']]


# In[24]:


#Data Analysis/ Exploration
df1.info()


# In[25]:


df1.head()


# In[26]:


df1.describe(include = 'all')  


# In[27]:


win_loss = df1[['WINorLOSS', 'FG%']].groupby('WINorLOSS')
fg = win_loss['FG%'].mean()
fg.plot(kind='bar')
plt.title('FG% in Win/Loss')
plt.ylabel('Shooting %')


# In[28]:


#Need to create a graph/table showing the percentage of games won for home games and percentage of games won for away games
home_advantage = df1[['Location', 'WINorLOSS']].groupby('Location')
home_advantage.describe(include = 'all')


# In[29]:


sns.kdeplot(df1['TeamPoints'], shade = True)
#Can see that it follow a normal distribution


# In[30]:


month_pts = df1[['Month', 'TeamPoints']]
month_pts = month_pts.groupby('Month').mean()
month_pts = pd.DataFrame(month_pts)
month_pts = month_pts.reindex(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])
month_pts


# In[31]:


month_pts.plot()
plt.ylabel('Average Team Points')
plt.title('Average Team Points Per Month')
plt.xticks([0, 1,2,3,4,5,6], ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])


# In[32]:


win_loss1 = df1[['WINorLOSS', 'FGA']]
win_loss1 = win_loss1.groupby('WINorLOSS').mean()
win_loss1 = pd.DataFrame(win_loss1)
win_loss1


# In[33]:


#Easily see there's a large difference between 3ptshot% with wins and losses
three_ball = df1[['WINorLOSS', '3ptShots', '3ptShotsAtt', '3ptShot%']]
three_ball = three_ball.groupby('WINorLOSS').mean() 
three_ball


# In[34]:


ft = df1[['WINorLOSS', 'FT', 'FTA', 'FT%']]
ft = ft.groupby('WINorLOSS').mean() 
ft


# In[35]:


rebounds = df1[['WINorLOSS', 'OREB', 'REB']]
rebounds = rebounds.groupby('WINorLOSS').mean() 
rebounds


# In[36]:


#can see there are more assists in teams that win vs teams that don't
rand = df1[['WINorLOSS', 'AST', 'STL', 'BLK', 'TO']]
rand = rand.groupby('WINorLOSS').mean() 
rand


# In[37]:


fouls= df1[['WINorLOSS', 'TotalFouls']]
fouls = fouls.groupby('WINorLOSS').mean() 
fouls


# In[38]:


#Important to create this dataframe within machine learning data frame
steal_to = df1[['WINorLOSS', 'AST', 'TO']]
steal_to['AST_TO'] = df1['AST']/df1['TO']
steal_to1 = steal_to[['WINorLOSS', 'AST_TO']]
steal_to1 = steal_to1.groupby('WINorLOSS')
steal_to1.mean()


# In[39]:


df1['AST_TO'] = df1['AST']/df1['TO']


# In[40]:


#Did not receive any good info using pairplots
corr= df1.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, annot = True, xticklabels = corr.columns, yticklabels = corr.columns, cmap="RdYlGn")
plt.show()
#Attributes shown with high correlations 
#WINorLOSS not associated with FGA, 3ptShotAtt, OREB, 
#Remove FG, OREB, Point_dif, FT


# In[41]:


df1 = df1.drop(columns= ['FG', 'OREB', 'Point_Dif', 'FT', '3ptShots', '3ptShotsAtt', 'FGA','TO'])


# In[42]:


corr= df1.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, annot = True, xticklabels = corr.columns, yticklabels = corr.columns, cmap="RdYlGn")
plt.show()
#Look with TeamPoints removed


# In[43]:


#Splitting the season into quarters
df1['Game'] = df1['Game'].astype(int)
df1.loc[df1['Game'] <= 20, 'Game'] = 0
df1.loc[(df1['Game'] > 20) & (df1['Game'] <= 41 ), 'Game'] = 1
df1.loc[(df1['Game'] > 41) & (df1['Game'] <= 61 ), 'Game'] = 2
df1.loc[(df1['Game'] > 61) & (df1['Game'] <= 82 ), 'Game'] = 3


# In[44]:


df1 = pd.get_dummies(df1, columns = ['Team', 'Year', 'Opponent', 'Game', 'Location', 'Month'], drop_first = True)


# In[45]:


df1.head()


# In[46]:


y = df1['WINorLOSS']
x = df1.drop('WINorLOSS', axis = 1)


# In[47]:


#Create the test data frame
x_train1, x_test, y_train1, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[48]:


#Need to implement feature scaling
from sklearn.preprocessing import StandardScaler
columns = ['TeamPoints','FG%', '3ptShot%', 'FTA', 'FT%', 'REB', 'AST', 'STL', 'BLK', 'TotalFouls', 'AST_TO']


# In[49]:


#FEature Scaling
scaler = StandardScaler()
x_train1[columns] = scaler.fit_transform(x_train1[columns])
x_test[columns] = scaler.transform(x_test[columns])


# In[50]:


x_train1.head()


# In[51]:


#Now we can start implementing machine learning models
#Linear Gradient Descent
sgd = SGDClassifier(max_iter = 500, tol = None)
sgd.fit(x_train1, y_train1)
sgd.score(x_train1, y_train1)
acc_sgd = round(sgd.score(x_train1, y_train1) * 100, 2)


# In[52]:


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train1, y_train1)
acc_logreg = round(logreg.score(x_train1, y_train1) * 100, 2)


# In[53]:


#Random Forest
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(x_train1, y_train1)
acc_random_forest = round(random_forest.score(x_train1, y_train1) * 100, 2)


# In[54]:


#K-Neighbors CLassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train1, y_train1)
acc_knn = round(knn.score(x_train1, y_train1) * 100, 2)


# In[55]:


#Guassian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train1, y_train1)
acc_gaussian = round(gaussian.score(x_train1, y_train1)* 100, 2)


# In[56]:


#Perceptron
perceptron = Perceptron(max_iter = 5)
perceptron.fit(x_train1, y_train1)
acc_perceptron = round(perceptron.score(x_train1, y_train1)*100, 2)


# In[57]:


#Linear Support Vector Machine
linear_svc = LinearSVC()
linear_svc.fit(x_train1, y_train1)
acc_linear_svc = round(linear_svc.score(x_train1, y_train1) *100, 2)


# In[58]:


#Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train1, y_train1)
acc_decision_tree = round(decision_tree.score(x_train1, y_train1)*100, 2)


# In[59]:


#Which model had the greatest accuracy?
scores_df = pd.DataFrame({'Model': ['Linear Gradient Descent', 'Logistic Regresstion', 'Random Forest', 'KNN', 'Gaussian Naive Bayes', 'Perceptron', 'Linear Support Vector Machines', 'Decision Trees'], 'Score': [acc_sgd, acc_logreg, acc_random_forest, acc_knn, acc_gaussian, acc_perceptron, acc_linear_svc, acc_decision_tree]})
scores1_df = scores_df.sort_values(by = 'Score', ascending = False)
scores1_df = scores1_df.set_index('Score')
scores1_df


# In[60]:


#Because Random forests is a collection of decision trees we will use the Random Forest model as our classifier. 
#The model has an accuracy score of 100% which is very alarming. Lets run some cross validation to see if we can fix this.
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators = 100)
scores = cross_val_score(rf, x_train1, y_train1, cv=5, scoring = 'accuracy')
print(scores)
print(scores.mean())


# In[61]:


#Model's average accuracy is 80.11%
#Lets find the feature importance of the models
importances = pd.DataFrame({'feature': x_train1.columns, 'importance':np.round(random_forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending = False)
importances


# In[62]:


#Forgot that the year was still in the model.
#In order ti predict results of a particular matchup the year in the model will not matter.
#ALso, the location of the game doesn't seem to matter as much either
#We'll drop both columns from our dataset
datasets = [x_train1, x_test]
for set in datasets:
    set.drop(['Year_2015', 'Year_2016', 'Year_2017', 'Year_2018', 'Location_Home', 'Game_1', 'Game_2', 'Game_3'], axis=1,  inplace = True)


# In[63]:


#Train the model with the dropped columns
rf = RandomForestClassifier(n_estimators=100, oob_score = True, random_state =1)
rf.fit(x_train1, y_train1)
rf_y_pred = rf.predict(x_test)
acc_rf = round(rf.score(x_train1, y_train1) * 100, 2)
acc_rf


# In[64]:


#Now Lets use cross validation to see if our accuracy has truly grown
scores = cross_val_score(rf, x_train1, y_train1, cv = 5, scoring = 'accuracy')
print(scores)
print(scores.mean())


# In[65]:


#Our accuracy basically, stayed the same.. this reduction will help us better generalize our machine learning algorthim. 
#Thus, the reduction truly benefits us. 
#In a random forest classifier the Out of Bag samples (OOB) meausres the accuracy 
#of predictions with the decision trees within the forest that that certain observation was not used to build.
oob = round(rf.oob_score_, 4) * 100
oob


# In[ ]:


#Our OOB score is slightly lower than the accuracy. We can possibly increase our accuracy with some hyperparameter tuning next.
#min_samples_split is the required amount of samples to be in each category in order to split the node
    #Example'2' would mean have 2 in short, tall and medium
#min_samples_leaf minimum number of samples to be at the leaf in order to categorize it as that type
#Finding the best parameters for the model
param_grid = { 'min_samples_split': [2, 5, 10, 25], 'min_samples_leaf': [ 1, 5, 10, 25, 50], 'n_estimators': [100, 300, 5000]}
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_estimators = 100, max_features = 'auto', oob_score=True, random_state=2, n_jobs = -1)

grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv=3)

grid.fit(x_train1, y_train1)


# In[ ]:


grid.best_params_


# In[66]:


#Now fit the new model with the best parameters
rf = RandomForestClassifier(min_samples_leaf = 1, min_samples_split = 2, n_estimators = 500, oob_score = True, random_state=4)
rf.fit(x_train1, y_train1)
y_pred_test = rf.predict(x_test)
print(round(rf.oob_score_, 4) *100)


# In[67]:


#Further Evaluation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


# In[68]:


#Evaluate the confusion matrix
#It seems as if the model is making accurate predictions accross the board
pred = cross_val_predict(rf, x_train1, y_train1, cv=3)
print(confusion_matrix(y_train1, pred))
print('Precision: ', precision_score(y_train1, pred))
print('Recall: ', recall_score(y_train1, pred))
print('F1 Score: ', f1_score(y_train1, pred))


# In[69]:


#ROC-AUC Score
y_scores = rf.predict_proba(x_test)
y_scores= y_scores[:,1]
fprate, tprate, thresh = roc_curve(y_test, y_scores)
plt.title('ROC AUC Curve')
plt.plot(fprate, tprate, 'b', linewidth=2)
plt.plot([0,1], [0,1], 'r', linewidth=4)
plt.axis([0,1,0,1])
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.figure(figsize=(14,7))
plt.show()
print('ROC-AUC Score: ', roc_auc_score(y_test, y_scores))
#Represents the TP rate vs the FP rate. Our model seems to do a decent job. There is a major tradeoff as the more false positives the higher the true positive rate
#Think of this tradef
#ALso, our ROC-AUC score is very high at 89%. 


# In[73]:


#Test the model against the test set
from sklearn.metrics import accuracy_score
print('Accuracy Score:', accuracy_score(y_test, y_pred_test))
print('F1 score: ', f1_score(y_test, y_pred_test))
print('Precision: ', precision_score(y_test, y_pred_test))
print('Recall: ', recall_score(y_test, y_pred_test))
print('ROC-AUC: ', roc_auc_score(y_test, y_pred_test))


# In[ ]:




