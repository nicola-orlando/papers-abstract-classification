# acaps-test
 Instructions for the ACAPS selection test 

## Installations

Please install the following libraries from your home folder.  

- Download python3, e.g., for mac https://www.python.org/downloads/macos/. I use Python 3.8. Remember to update your library path
        
        PATH='/Users/your_username/Library/Python/X.XX/bin/:'$PATH 

- Pandas 

        pip3 install pandas
        
- NumPy 

        pip install numpy 

- Install Scikit-learn (https://scikit-learn.org/stable/install.html)

        pip install -U scikit-learn 

- Install matplotlib (https://matplotlib.org/stable/users/installing/index.html)

        pip install matplotlib

- Install Seaborn (https://seaborn.pydata.org/installing.html)

        pip install seaborn

## Data

- Original data from: https://www.kaggle.com/datasets/abisheksudarshan/topic-modeling-for-research-articles
- Data used in this repo is collected under the 'data' folder. Obtained via private correspondence 

## Download and run the code 

    git clone git@github.com:nicola-orlando/acaps-test.git 
    cd acaps-test
    python3 nlp_model.py

## Expected results 

Mean labels distribution per topic to investigate the samples population

![topics](https://user-images.githubusercontent.com/26884030/207294639-5036ebb5-4f32-4651-bd67-2026711477ed.png)
![research_areas](https://user-images.githubusercontent.com/26884030/207294707-4e3285e5-63ee-4cac-8992-343e705f6cb3.png)

The distribution of the topics will not be used further. The Distribution of the research areas (also referred to as 'topics') represent the set of classes I to predict. 
The plots show that the training data has a non-uniform distribution of the classes. Specifically, the target class ("Statistics"), has only about 15% of the research papers belonging to it.  

The model main model I trained is a logistic regression model with default parameters using the "ABSTRACTS" as features with corpus modelled according to the Term Frequency - Inverse Document Frequency (TF-IDF) model. 
Alternatively I trained an SVC classifier and a MultinomialNB model with alpha=1. They have similar level of performance for the used metrics. Modelling the corpus with simple tokens counting leads to sub-optimal results, this option is not discussed here. 

The results of some performance metrics are here (target class is "Statistics"):

| Model                   | ROC AUC       | Accuracy  | F1 score |
| -------------                |:-------------:|:-----:| -----:|
| LogisticRegression (**Default**) | 0.838 | 0.874 | 0.794  |
| SVC                              | 0.718 | 0.896 | 0.762  |  
| MultinomialNB, alpha=1           | 0.740 | 0.897 | 0.776  |


Finally, to characterise the performance of the logistic regression model the code will produce some example of plots, as showed below. All plots are produced assuming "Statistics" to be the topic to be identified. 

![precision_recall_Statistics](https://user-images.githubusercontent.com/26884030/207298733-740f4f02-9d39-488c-9da2-12817ef99a3e.png)
![roc_curve_Statistics](https://user-images.githubusercontent.com/26884030/207298749-e88d3a33-c884-4c07-8642-f54ef81a327f.png)
![heatmap_Statistics](https://user-images.githubusercontent.com/26884030/207298716-1bfe1ebe-a477-4447-86b5-10ef004585d9.png)
