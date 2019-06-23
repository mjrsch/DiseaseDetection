import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
cancerSamplesCoimbra = pd.read_csv('Breast_Cancer_Coimbra.csv')
colsCoimbra = [column for column in cancerSamplesCoimbra.columns if column not in ['Classification']]
dataCoimbra = cancerSamplesCoimbra[colsCoimbra]
targetCoimbra = cancerSamplesCoimbra['Classification']

#split data set into train and test sets
data_train_coimbra, data_test_coimbra, target_train, target_test = train_test_split(dataCoimbra,targetCoimbra, test_size = 0.30, random_state = 10)


# Naive-Bayes
#from sklearn.naive_bayes import GaussianNB
def naiveBayesCoimbra():
    from sklearn.naive_bayes import GaussianNB
    gnb_coimbra = GaussianNB()
    pred_gnb_coimbra = gnb_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    print("Naive-Bayes accuracy Coimbra: ",accuracy_score(target_test, pred_gnb_coimbra, normalize = True))
    return gnb_coimbra

def getNaiveBayesCoimbraAccuracy():
    from sklearn.naive_bayes import GaussianNB
    gnb_coimbra = GaussianNB()
    pred_gnb_coimbra = gnb_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    return accuracy_score(target_test, pred_gnb_coimbra, normalize = True)


#SVM
def supportVectorMachinesCoimbra():
    from sklearn.svm import SVC
    svc_coimbra = SVC(kernel = 'linear', random_state = 0)
    pred_svc_coimbra = svc_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    print('SVC Classifier accuracy Coimbra:', accuracy_score(target_test, pred_svc_coimbra, normalize = True))
    return svc_coimbra

def getSupportVectorMachinesCoimbraAccuracy():
    from sklearn.svm import SVC
    svc_coimbra = SVC(kernel = 'linear', random_state = 0)
    pred_svc_coimbra = svc_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    return accuracy_score(target_test, pred_svc_coimbra, normalize = True)


#K-Nearest Neighbor
def kNearestNeigborCoimbra():
    from sklearn.neighbors import KNeighborsClassifier
    knn_coimbra = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    pred_knn_coimbra = knn_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    print('K-Neighbors accuracy Coimbra:', accuracy_score(target_test, pred_knn_coimbra, normalize = True))
    return knn_coimbra

def getKNearestNeigborCoimbraAccuracy():
    from sklearn.neighbors import KNeighborsClassifier
    knn_coimbra = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    pred_knn_coimbra = knn_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    return accuracy_score(target_test, pred_knn_coimbra, normalize = True)


#Decision Tree 
def decisionTreeCoimbra():
    from sklearn.tree import DecisionTreeClassifier
    decisionTree_coimbra = DecisionTreeClassifier(criterion = 'entropy', random_state = 10)
    pred_decisionTree_coimbra = decisionTree_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    print('Decision Tree accuracy Coimbra:', accuracy_score(target_test, pred_decisionTree_coimbra, normalize = True))
    return decisionTree_coimbra

def getDecisionTreeCoimbraAccuracy():
    from sklearn.tree import DecisionTreeClassifier
    decisionTree_coimbra = DecisionTreeClassifier(criterion = 'entropy', random_state = 10)
    pred_decisionTree_coimbra = decisionTree_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    return accuracy_score(target_test, pred_decisionTree_coimbra, normalize = True)


#Fitting Random Forest Classification Algorithm
def randomForestCoimbra():
    from sklearn.ensemble import RandomForestClassifier
    randomForest_coimbra = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 10)
    pred_randomForest_coimbra = randomForest_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    print('Random Forest accuracy Coimbra:', accuracy_score(target_test, pred_randomForest_coimbra, normalize = True))
    return randomForest_coimbra

def getRandomForestCoimbraAccuracy():
    from sklearn.ensemble import RandomForestClassifier
    randomForest_coimbra = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 10)
    pred_randomForest_coimbra = randomForest_coimbra.fit(data_train_coimbra, target_train).predict(data_test_coimbra)
    return accuracy_score(target_test, pred_randomForest_coimbra, normalize = True)

