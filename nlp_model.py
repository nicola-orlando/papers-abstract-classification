# General 
import numpy as np
import nltk
import pandas as pd

# Topic modelling 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ML model building  
from sklearn.model_selection import train_test_split
# Ameba 
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix 

train_df = pd.read_csv('data/Train.csv')
print(train_df.head())

test_df = pd.read_csv('data/Test.csv')
print(test_df.head())

# Let's check now the available topics
topics_df = pd.read_csv('data/Tags.csv')
print(topics_df.head(-1))

# Train and test data have different columns. The dataset has only the final categories, while the training data also includes the topics.
# Overall we have 2 columns for ID and abstract, 25 columns for the topics and 4 for the categories of publications. 

# Check the size of the dataframe
print('Print number of rows in Train.csv = '+str(len(train_df.index)))

# Get the column names
names_columns = np.array(train_df.columns)
print('Available columns in Train.csv (total of %i)'%(len(names_columns)))
print(names_columns)
# The first two columns are ID and title, the remaining one are lables I would want to eventually predict

# Now we use the list of columns for two tasks:
# 1- check if there are Nans in the dataset 
# 2- Check the average values of the labels. This is relevant check if the dataset is balanced or not
average_labels = []
for i in range(0,len(names_columns)):
    # Print that there are Nans only if we find some
    num_of_nans = train_df[names_columns[i]].isnull().sum()
    if num_of_nans > 0: 
        print('Number of nans is '+str(num_of_nans))
    
    # if we are in one of the labels columns check their average. 
    # where 0.5 is the average label value for a single label dataset that is also balanced balanced 
    if i >=2 :
        # print(names_columns[i]+': label average= '+str(train_df[names_columns[i]].mean()))
        average_labels.append(train_df[names_columns[i]].mean())

# Basic plots
import matplotlib.pyplot as plt

def make_bar_chart(x, y, title, plot_name): 
    fig = plt.figure(figsize = (8, 6))
    plt.bar(x, y, color='lavender', width=0.6, linewidth=0.5, edgecolor='black')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid(True,axis='y')
    plt.savefig(plot_name)

make_bar_chart(names_columns[2:6],average_labels[:4],'Average label values per research area','research_areas.png')
make_bar_chart(names_columns[6:],average_labels[4:],'Average label values per research topic','topics.png')

# From what we can see, there are no Nans in the dataset, but we have an unbalanced dataset.
# The category 'Computer Science' has the largest average, of about 0.4, the lowest average is about 2 times lower (Mathematics). Overall we can't expect the classes to be balanced 

# Train the model
# Here I will train a model using the abstracts and the basic classification given by names_columns[2:6]

# I want to check how well the model does in predicting the label 'Astrophysics of Galaxies' so I need to use part of the training dataset to be able to access the relevant topic label.
# The topic labels are not avaiable in the given Train.csv file. 
# X_train, X_test, y_train, y_test = train_test_split(train_df[['ABSTRACT','Computer Science', 'Physics', 'Mathematics', 'Statistics']],
X_train, X_test, y_train, y_test = train_test_split(train_df[['ABSTRACT','Computer Science', 'Physics', 'Mathematics', 'Statistics']], 
                                                    train_df['Astrophysics of Galaxies'], 
                                                    random_state=0)

print('inspecting types from train_test_split')
print(type(X_train))

# Here adding the other remaining features available at training and testing time which represent a basic classification of the abstracts (names_columns[2:6])
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

vectorizer = TfidfVectorizer()

# Incrementally add the columns to the sparse feature matrix for both training and testing set. 
X_vectorised = vectorizer.fit_transform(X_train['ABSTRACT'])
X_vec_1 = add_feature(X_vectorised, X_train['Computer Science'])
X_vec_2 = add_feature(X_vec_1, X_train['Mathematics'])
X_vec_3 = add_feature(X_vec_2, X_train['Physics'])
X_vec_fin = add_feature(X_vec_3, X_train['Statistics'])
#
X_test_vectorised = vectorizer.transform(X_test['ABSTRACT'])
X_test_vec_1 = add_feature(X_test_vectorised, X_test['Computer Science'])
X_test_vec_2 = add_feature(X_test_vec_1, X_test['Mathematics'])
X_test_vec_3 = add_feature(X_test_vec_2, X_test['Physics'])
X_test_vec_fin = add_feature(X_test_vec_3, X_test['Statistics'])


# Ameba 
#cls = MultinomialNB(alpha=0.1)

# Define a model with balanced classes and perform a basic HP scan. Can be made more complicated as needed. 
cls = LogisticRegression(random_state=0, class_weight='balanced')
parameters = {'C': [0.5,1,1.5]}
clf = GridSearchCV(cls, param_grid=parameters, scoring='roc_auc')  
clf.fit(X_vec_fin, y_train)
y_predict = clf.best_estimator_.predict(X_test_vec_fin)

# Evaluate a number of possible metrics 
roc_auc = roc_auc_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
confusion_matrix = confusion_matrix(y_test,y_predict)

print('Final metrics roc_auc = %f, recall = %f, precision = %f, accuracy =%f'%(roc_auc,recall,precision,accuracy))
print('Confusion matrix')
print(confusion_matrix)

import seaborn as sn
fig = plt.figure(figsize = (4, 4))
plt.title('Confusion matrix')
ax = sn.heatmap(confusion_matrix, cmap='Oranges', annot=True)
plt.tight_layout()
plt.savefig('heatmap.png')
