import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


class LogisticRegressionBinary:
    def __init__(self):
        self.regressor = None

    def train_validation(self, train_x, train_y, max_iter=100):
        
        # Faz uma Cross Validation com 10 valores de C e 5 folds para descobrir o melhor C
        self.regressor = LogisticRegressionCV(Cs=10, cv=5, max_iter=max_iter)
        self.regressor.fit(train_x, train_y)

        ypred = self.regressor.predict(train_x)

        accuracy = accuracy_score(train_y, ypred)
        precision = precision_score(train_y, ypred)
        f1 = f1_score(train_y, ypred)
        cm = confusion_matrix(train_y, ypred)

        return accuracy, precision, f1
    
    def test(self, test_x, test_y):
        
        if self.regressor == None:
            print("Treine o regressor! Use o método train_validation(self, train_x, train_y, max_iter=100)")
            return 

        ypred = self.regressor.predict(test_x)

        accuracy = accuracy_score(test_y, ypred)
        precision = precision_score(test_y, ypred)
        f1 = f1_score(test_y, ypred)
        cm = confusion_matrix(test_y, ypred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        class_labels = ['NH', 'H']
        ticks = np.arange(len(class_labels))

        plt.title('Matriz de Confusão Normalizada do Regressor Binário')
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(ticks, class_labels)
        plt.yticks(ticks, class_labels)

        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, format(cm_normalized[i, j], '.2f'), horizontalalignment="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")

        plt.xlabel('Classe predita')
        plt.ylabel('Classe verdadeira')
        plt.tight_layout()

        plt.savefig("./gráficos/LR/confusionMatrixBinaryTest.png")

        return accuracy, precision, f1

class LogisticRegressionMulticlass:
    def __init__(self):
        self.regressor = None

    def train_validation(self, train_x, train_y, max_iter=100):
        
        # Faz uma Cross Validation com 10 valores de C e 5 folds para descobrir o melhor C
        self.regressor = LogisticRegressionCV(Cs=10, cv=5, max_iter=max_iter)
        self.regressor.fit(train_x, train_y)

        ypred = self.regressor.predict(train_x)

        accuracy = accuracy_score(train_y, ypred)
        precision = precision_score(train_y, ypred, average='macro')
        f1 = f1_score(train_y, ypred, average='macro')
        cm = confusion_matrix(train_y, ypred)

        return accuracy, precision, f1
    
    def test(self, test_x, test_y):
        
        if self.regressor == None:
            print("Treine o regressor! Use o método train_validation(self, train_x, train_y, max_iter=100)")
            return 

        ypred = self.regressor.predict(test_x)

        accuracy = accuracy_score(test_y, ypred)
        precision = precision_score(test_y, ypred, average='macro')
        f1 = f1_score(test_y, ypred, average='macro')
        cm = confusion_matrix(test_y, ypred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        class_labels = ['A', 'D', 'H']
        ticks = np.arange(len(class_labels))

        plt.title('Matriz de Confusão Normalizada do Regressor')
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(ticks, class_labels)
        plt.yticks(ticks, class_labels)

        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, format(cm_normalized[i, j], '.2f'), horizontalalignment="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")

        plt.xlabel('Classe predita')
        plt.ylabel('Classe verdadeira')
        plt.tight_layout()

        plt.savefig("./gráficos/LR/confusionMatrixMulticlassTest.png")

        return accuracy, precision, f1








