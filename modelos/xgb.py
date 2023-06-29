import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, make_scorer
from sklearn.model_selection import GridSearchCV

class XGB():

    def train_validation(self, x_train, y_train):
        parameters = {
            'learning_rate': [0.1, 0.01],
            'n_estimators': [40, 80],
            'max_depth': [3, 15],
            'min_child_weight': [3, 10, 20],
            'gamma': [0.4],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'scale_pos_weight': [1],
            'reg_alpha': [1e-5]
        }
        # botei pouca variedade de valores pra ficar mais rápido mesmo

        clf = xgb.XGBClassifier(seed=2)
        f1_scorer = make_scorer(f1_score, pos_label=1)

        grid_obj = GridSearchCV(
            clf,
            scoring=f1_scorer,
            param_grid=parameters,
            cv=5,
            verbose=2
        )

        grid_obj = grid_obj.fit(x_train, y_train)
        best_params = grid_obj.best_params_

        return best_params


    def test(self, x_train, y_train, x_test, y_test, best_params):
        clf = xgb.XGBClassifier(**best_params)

        clf.fit(x_train, y_train)

        y_pred_train = clf.predict(x_train)
        f1_train = f1_score(y_train, y_pred_train, pos_label=1)
        acc_train = accuracy_score(y_train, y_pred_train)

        print("F1 score and accuracy score for training set: {:.4f}, {:.4f}.".format(f1_train, acc_train))

        y_pred_test = clf.predict(x_test)
        f1_test = f1_score(y_test, y_pred_test, pos_label = 1)
        acc_test = accuracy_score(y_test, y_pred_test)
        prec_test = precision_score(y_test, y_pred_test, pos_label = 1)


        print("F1 score and accuracy score for test set: {:.4f}, {:.4f}.".format(f1_test, acc_test))

        cm = confusion_matrix(y_test, y_pred_test)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        labels = ['NH', 'H']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Valor predito - Target')
        plt.ylabel('Valor real - Target')
        plt.title('Matriz de Confusão Normalizada - XGB')
        plt.savefig('./gráficos/XGB/confusionMatrixTest.png')
        plt.show()

        y_pred = clf.predict(x_train)
        f1 = f1_score(y_train, y_pred, pos_label = 1)
        acc = accuracy_score(y_train, y_pred)
        prec  = precision_score(y_train, y_pred, pos_label = 1)

        return acc, prec, f1, acc_test, prec_test, f1_test
