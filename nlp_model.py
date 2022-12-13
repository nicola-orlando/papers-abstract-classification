# General 
import numpy as np
import pandas as pd

# Topics representation. Eventually I used the TF-IDF model.  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# A few models I tried. Eventually I used the LogisticRegression model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Models evaluation metrics 
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score 
from sklearn.metrics import roc_curve 
from sklearn.metrics import precision_recall_curve

# Basic plots
import matplotlib.pyplot as plt
import seaborn as sn

train_df = pd.read_csv('data/Train.csv')
print(train_df.head())

test_df = pd.read_csv('data/Test.csv')
print(test_df.head())

# Let's check now the available topics
topics_df = pd.read_csv('data/Tags.csv')
print(topics_df.head(-1))

# Train and test data have different columns. The dataset has only the final categories, while the training data also includes the topics.
# Overall we have 2 columns for ID and abstract, 25 columns for the topics and 4 for the categories of publications. The latter is what we will use along with the abstract. 

# Check the size of the dataframe
print('Print number of rows in Train.csv = '+str(len(train_df.index)))

# Get the column names
names_columns = np.array(train_df.columns)
print('Available columns in Train.csv (total of %i)'%(len(names_columns)))
# The first two columns are ID and title, the remaining one are lables I would want to eventually predict. The rest only uses columns from 2 to 6. 

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

# Define some plotting functions I will use below

def conf_matrix_plot(confusion_matrix, plot_name):
    fig = plt.figure(figsize = (5, 5))
    plt.title('Confusion matrix, class '+tclass)
    ax = sn.heatmap(confusion_matrix, cmap='Oranges', annot=True)
    plt.tight_layout()
    plt.savefig('heatmap_'+plot_name+'.png')
    
def roc_curve_plot(fpr,tpr,plot_name):
    fig = plt.figure(figsize = (5, 5))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr,tpr,linewidth=2)
    plt.plot([0,1],[0,1],'k--')
    plt.tight_layout()
    plt.grid(True,axis='y')
    plt.savefig('roc_curve_'+plot_name+'.png')

def precision_recall_plot(precision,recall,plot_name):
    fig = plt.figure(figsize = (5, 5))
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.plot(precision,recall,linewidth=2)
    plt.tight_layout()
    plt.grid(True,axis='y')
    plt.savefig('precision_recall_'+plot_name+'.png')
    
# Now let's train a model, say to distinguish Statistics articles from the test 

list_of_target_classes = ['Statistics']

for tclass in list_of_target_classes: 

    X_train = train_df['ABSTRACT']
    y_train = train_df[tclass]
    X_test = test_df['ABSTRACT']
    y_test = test_df[tclass]
    
    vectorizer = TfidfVectorizer()
    X_vectorised = vectorizer.fit_transform(X_train)
    X_test_vectorised = vectorizer.transform(X_test)

    # Here I use a LogisticRegression out of the box
    cls = LogisticRegression(random_state=0, class_weight='balanced')
    cls.fit(X_vectorised, y_train)
    y_predict = cls.predict(X_test_vectorised)
    
    # Evaluate a number of possible metrics 
    roc_auc = roc_auc_score(y_test, y_predict, average='macro')
    accuracy = accuracy_score(y_test, y_predict)
    f1_score = f1_score(y_test, y_predict, average='macro')

    confusion_matrix = confusion_matrix(y_test,y_predict)

    print('Final metrics roc_auc = %f, accuracy =%f, f1_score = %f'%(roc_auc,accuracy,f1_score))
    print('Confusion matrix')
    print(confusion_matrix)

    conf_matrix_plot(confusion_matrix, tclass)
    
    fpr, tpr, thresholds = roc_curve(y_test,y_predict)
    roc_curve_plot(fpr, tpr, tclass)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test,y_predict)
    precision_recall_plot(precisions,recalls,tclass)    
