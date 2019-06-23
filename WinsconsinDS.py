
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancerSamples = pd.read_csv('Health_Breast_Cancer_Winsconsin.csv')
cols = [col for col in cancerSamples.columns if col not in ['id','diagnosis']]
data = cancerSamples[cols]
target = cancerSamples['diagnosis']

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

#feature selection for the Wisconsin DataSet
from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=31)
select.fit(data_train, target_train)
data_train_feature_selected = select.transform(data_train)
data_test_feature_selected =  select.transform(data_test)
print('X_train.shape is: {}'.format(data_train.shape))
print('X_train_selected.shape is: {}'.format(data_train_feature_selected.shape))
mask = select.get_support()
print(mask)



# Naive-Bayes
def naiveBayesWinsconsin():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    pred_gnb = gnb.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred_gnb, normalize = True))
    return gnb


def getNaiveBayesWinsconsinAccuracy():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    pred_gnb = gnb.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    return accuracy_score(target_test, pred_gnb, normalize = True)

#SVM
def supportVectorMachinesWinsconsin():
    from sklearn.svm import SVC
    svc = SVC(kernel = 'linear', random_state = 0)
    pred_svc = svc.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    print('SVC Classifier accuracy:', accuracy_score(target_test, pred_svc, normalize = True))
    return svc

def getSupportVectorMachinesWinsconsinAccuracy():
    from sklearn.svm import SVC
    svc = SVC(kernel = 'linear', random_state = 0)
    pred_svc = svc.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    return accuracy_score(target_test, pred_svc, normalize = True)



#K-Nearest Neighbor
def kNearestNeigborWinsconsin():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    pred_knn = knn.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    print('K-Neighbors accuracy :', accuracy_score(target_test, pred_knn, normalize = True))
    return knn

def getKNearestNeigborWinsconsinAccuracy():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    pred_knn = knn.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    return accuracy_score(target_test, pred_knn, normalize = True)


#Decision Tree 
def decisionTreeWinsconsin():
    from sklearn.tree import DecisionTreeClassifier
    decisionTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 10)
    pred_decisionTree = decisionTree.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    print('Decision Tree accuracy :', accuracy_score(target_test, pred_decisionTree, normalize = True))
    return decisionTree

def getDecisionTreeWinsconsinAccuracy():
    from sklearn.tree import DecisionTreeClassifier
    decisionTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 10)
    pred_decisionTree = decisionTree.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    return accuracy_score(target_test, pred_decisionTree, normalize = True)


#Fitting Random Forest Classification Algorithm
def randomForestWinsconsin():
    from sklearn.ensemble import RandomForestClassifier
    randomForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 10)
    pred_randomForest = randomForest.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    print('Random Forest accuracy :', accuracy_score(target_test, pred_randomForest, normalize = True))
    return randomForest

def getRandomForestWinsconsinAccuracy():
    from sklearn.ensemble import RandomForestClassifier
    randomForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 10)
    pred_randomForest = randomForest.fit(data_train_feature_selected, target_train).predict(data_test_feature_selected)
    return accuracy_score(target_test, pred_randomForest, normalize = True)

