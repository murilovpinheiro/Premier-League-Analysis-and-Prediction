import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import matplotlib.pyplot as plt
import seaborn as sns

def classes_resultado(x):
  if x > 0.5: return 2
  elif x > -0.5: return 1
  return 0
  
 ########## CLASSIFICAÇÃO ##########
# Usando classificação
# o y precisa apenas do resultado das partidas

class RandomForestBinary():
    
  def train_validation(self, X_train, y_train, goalsTrain):
      param_grid = {
          'n_estimators': [10, 50, 100,200],
          'max_depth': [5, 50],
          "max_leaf_nodes": [None, 10, 50]
      }

      print("Começando validação...")
      
      model = RandomForestRegressor()
      cv = KFold(n_splits=5)
      grid_search1 = GridSearchCV(model, param_grid, cv=cv, n_jobs = 5)
      grid_search1.fit(X_train, goalsTrain)
      
      print("Validação terminada.")
      
      best_parameters = grid_search1.best_params_
      print("Melhores parametros:", best_parameters)
      return best_parameters
    
  def test(self, X_train, y_train, X_test, y_test, goalsTrain, goalsTest, parameters = None):
      rf = RandomForestRegressor(**parameters)

      print("Treinando modelo...")
      rf.fit(X_train, goalsTrain)

      train_pred = rf.predict(X_train)

      train_pred_dif = train_pred[:,0] - train_pred[:,1]
      train_pred = (train_pred_dif > 0.5 ).astype(int)

      test_pred = rf.predict(X_test)
      test_pred_dif = test_pred[:,0] - test_pred[:,1]
      test_pred = (test_pred_dif > 0.5 ).astype(int)

      train_accuracy = accuracy_score(y_train, train_pred)
      train_f1 = f1_score(y_train, train_pred)
      train_prec = precision_score(y_train, train_pred)
      
      test_accuracy = accuracy_score(y_test, test_pred)
      test_f1 = f1_score(y_test, test_pred)
      test_prec = precision_score(y_test, test_pred)
      
      cm = confusion_matrix(y_test, test_pred, normalize = "true")
      sns.heatmap(cm, annot=True, fmt='f', cmap='Blues')
      plt.xlabel('Rótulo Predito')
      plt.ylabel('Rótulo Real')
      plt.title('Matriz de Confusão - RandomForest')
      plt.savefig("./gráficos/RF/Test_CM.png")

      print(train_accuracy, test_accuracy)
      
      return train_accuracy, train_prec, train_f1, test_accuracy, test_prec, test_f1 
    
 ########## REGRESSÃO ##########
 # Usando regressão para os gols
 # O y precisa dos HomeGoals e AwayGoals

class RandomForestMulticlass():
  
  def train_validation(self, X_train, y_train, goalsTrain):
      param_grid = {
          'n_estimators': [10, 50, 100,200],
          'max_depth': [5, 50],
          "max_leaf_nodes": [None, 10, 50]
      }

      print("Começando validação...")
      
      model = RandomForestClassifier()
      cv = KFold(n_splits=5)
      grid_search1 = GridSearchCV(model, param_grid, cv=cv, n_jobs = 10)
      grid_search1.fit(X_train, goalsTrain)
      
      print("Validação terminada.")
      
      best_parameters = grid_search1.best_params_
      print("Melhores parametros:", best_parameters)
      return best_parameters
      
  def test(self, X_train, y_train, X_test, y_test, goalsTrain, goalsTest, parameters = None):
      if parameters == None:
          depth = 6
          estimators = 40
      else:
          depth = parameters["max_depth"]
          estimators = parameters["n_estimators"]
      rf = RandomForestClassifier(n_estimators = estimators, max_depth = depth)
      
      def limiar(x):
        if x > 0.5: return 2
        elif x > -0.5: return 0
        return 1

      print("Treinando modelo...")
      rf.fit(X_train, goalsTrain)

      train_pred_reg = rf.predict(X_train)
      train_pred_dif = train_pred_reg[:,0] - train_pred_reg[:,1]
      train_pred = np.vectorize(limiar)(train_pred_dif)
      y_train = np.argmax(y_train, axis = 1)

      train_accuracy = accuracy_score(y_train, train_pred)
      train_prec = precision_score(y_train, train_pred, average = "macro")
      train_f1 = f1_score(y_train, train_pred, average = "macro")

      test_pred_reg = rf.predict(X_test)
      test_pred_dif = test_pred_reg[:,0] - test_pred_reg[:,1]
      test_pred = np.vectorize(limiar)(test_pred_dif)
      y_test = np.argmax(y_test, axis = 1)

      test_accuracy = accuracy_score(y_test, test_pred)
      test_prec = precision_score(y_test, test_pred, average = "macro")
      test_f1 = f1_score(y_test, test_pred, average = "macro")
      
      cm = confusion_matrix(y_test, test_pred, normalize = "true")
      sns.heatmap(cm, annot=True, fmt='f', cmap='Blues')
      plt.xlabel('Rótulo Predito')
      plt.ylabel('Rótulo Real')
      plt.savefig("./gráficos/RF/RFR_Multi_CM.png")

      print(train_accuracy, test_accuracy)
      
      return train_accuracy, train_prec, train_f1, test_accuracy, test_prec, test_f1