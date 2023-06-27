import numpy as np

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from datetime import datetime as dt

class Multiclass(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super().__init__()
        
        # utilizando ELU como função de ativação
        self.act = nn.ELU()
        self.outputact = nn.Softmax(dim = 1)
        
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.hidden4 = nn.Linear(128, 256)
        self.hidden5 = nn.Linear(256, 256)
        
        self.output = nn.Linear(256, output_size)
        
        self.dropout = nn.Dropout(dropout_rate)  # camada droupout é um tipo de regularização da rede neural
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.act(x)
        x = self.dropout(x)  # Aplicar Dropout após a primeira camada oculta

        x = self.hidden2(x)
        x = self.act(x)
        x = self.dropout(x)  

        x = self.hidden3(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.hidden4(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.hidden5(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.output(x)
        x = self.outputact(x)
        return x


def trainValidation(train_x, train_y):
    x_train = torch.tensor(train_x).float()
    y_train = torch.from_numpy(train_y).float()

    # Dividir o conjunto de treinamento em treinamento e teste pelo tempo
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3) 

    # Exibir o shape dos conjuntos de treinamento e validação
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_valid shape:", x_valid.shape)
    print("y_valid shape:", y_valid.shape)

    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    n_epochs = 200 # Queria mais mas acho que 200 tá bom

    momentums = [0.7, 0.8, 0.9]
    learning_rates = [0.1, 0.01, 0.001]
    weight_decays = [0.001, 0.0001, 0.00005] 

    best_acc = - np.nan
    models = []
    final_loss = []

    CROSS_losses_train = []
    CROSS_losses_val = []

    ACC_losses_train = []
    ACC_losses_val = []

    #PREC_losses_train = []
    #PREC_losses_val = []

    parameters = []
    batch_size = 64

    for weight_decay in weight_decays:
            for momentum in momentums:
                    for lr in learning_rates:
                            parameters.append({"learning_rate": lr,
                                            "momentum": momentum, 
                                            "weight_decay": weight_decay})
                                    
                            batches_per_epoch = len(x_train) // batch_size
                            model = Multiclass(input_size, output_size, 0.2)
                                    
                            current_CROSS_train = []
                            current_CROSS_val = []

                            current_ACC_train = []
                            current_ACC_val = []
                                    
                            loss_fn = nn.CrossEntropyLoss()
                            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = weight_decay, momentum = momentum)
                            for epoch in range(n_epochs):
                                    epoch_loss = []
                                    epoch_acc = []
                                    model.train()
                                    for i in range(batches_per_epoch):
                                            start = i * batch_size
                                            X_batch = x_train[start:start+batch_size]
                                            y_batch = y_train[start:start+batch_size]

                                                    # forward 
                                            y_pred = model(X_batch)
                                            loss = loss_fn(y_pred, y_batch)
                                                    # backward 
                                            optimizer.zero_grad()
                                            loss.backward()
                                                    # update weights
                                            optimizer.step()
                                            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                                            epoch_loss.append(float(loss))
                                            epoch_acc.append(float(acc))
                                            # set model in evaluation mode and run through the test set
                                    
                                    model.eval()
                                    y_pred = model(x_valid)
                                    print(y_pred.mean(axis = 0))
                                    print(y_valid.mean(axis = 0))
                                    ce = loss_fn(y_pred, y_valid)
                                    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_valid, 1)).float().mean()

                                    ce = float(ce)
                                    acc = float(acc)
                                    print(f"Epoch {epoch} train: Cross-entropy={sum(epoch_loss)/batches_per_epoch}, Accuracy={sum(epoch_acc)/batches_per_epoch}")
                                    
                                    current_CROSS_train.append(np.mean(epoch_loss))
                                    current_ACC_train.append(np.mean(epoch_acc))
                                    
                                    current_CROSS_val.append(ce)
                                    current_ACC_val.append(acc)
                                    
                                    if acc > best_acc:
                                            best_acc = acc
                                            best_weights = copy.deepcopy(model.state_dict())
                                    print(f"Epoch {epoch} validation: Cross-entropy={ce}, Accuracy={acc}")
                                    
                                    CROSS_losses_train.append(current_CROSS_train)
                                    CROSS_losses_val.append(current_CROSS_val)

                                    ACC_losses_train.append(current_ACC_train)
                                    ACC_losses_val.append(current_ACC_val)
                                    final_loss.append(ce)
    #menor_valor = min(final_loss)  # Encontra o menor valor na lista
    #indice_menor = final_loss.index(menor_valor)
    indice_menor = 0

    print(final_loss)

    best_CROSS_train = CROSS_losses_train[indice_menor]
    best_CROSS_val = CROSS_losses_val[indice_menor]
                
    best_ACC_train = ACC_losses_train[indice_menor]
    best_ACC_val = ACC_losses_val[indice_menor]

    print(parameters[indice_menor])
    parameters_ = parameters[indice_menor]

    epochs = range(1, len(CROSS_losses_train) + 1)

    fig, ax = plt.subplots(figsize = (10, 8))
    ax.plot(epochs, best_CROSS_train, 'b', label='Train Cross-Entropy')
    ax.plot(epochs, best_CROSS_val, 'r', label='Validation Cross-Entropy')
    plt.title('Cross-Entropy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Value')
    plt.legend()
    ax.grid(True)
    fig.savefig("./CEValidation.png")

    epochs = range(1, len(ACC_losses_train) + 1)
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.plot(epochs, best_ACC_train, 'b', label='Train Acc')
    ax.plot(epochs, best_ACC_val, 'r', label='Validation Acc')
    plt.title('Acc per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Value')
    plt.legend()
    ax.grid(True)
    fig.savefig("./ACCValidation.png")

    return parameters_



def trainTest(x_train, y_train, x_test, y_test, parameters):
    x_train = torch.tensor(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    x_test = torch.tensor(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Exibir o shape dos conjuntos de treinamento e validação
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    n_epochs = 300

    momentum = parameters["momentum"]
    learning_rate = parameters["learning_rate"]
    weight_decay = parameters["weight_decay"]

    best_acc = - np.nan
    final_loss = []

    batch_size = 64

    batches_per_epoch = len(x_train) // batch_size
    model = Multiclass(input_size, output_size, 0.2)

    CROSS_train = []
    CROSS_test = []

    ACC_train = []
    ACC_test = []

    n_epochs = 300

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay, momentum = momentum)
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        model.train()
        
        for i in range(batches_per_epoch):
            # Treinamento
            start = i * batch_size
            X_batch = x_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            # Forward
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Cálculo da acurácia e perda do treinamento
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))

        # Cálculo da acurácia e perda no conjunto de teste
        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_acc = (torch.argmax(y_pred_test, 1) == torch.argmax(y_test, 1)).float().mean()
        
        # Registro das métricas
        CROSS_train.append(np.mean(epoch_loss))
        ACC_train.append(np.mean(epoch_acc))
        CROSS_test.append(float(test_loss))
        ACC_test.append(float(test_acc))
        
        # Atualização dos melhores pesos
        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
        
        # Impressão das métricas
        print(f"Epoch {epoch} Train: Cross-entropy={np.mean(epoch_loss)}, Accuracy={np.mean(epoch_acc)}")
        print(f"Epoch {epoch} Test: Cross-entropy={float(test_loss)}, Accuracy={float(test_acc)}")
        final_loss.append(float(test_loss))

    epochs = range(1, len(CROSS_train) + 1)
    fig, ax = plt.subplots(figsize = (10, 8))  
    ax.plot(epochs, CROSS_train, 'b', label='Train Cross-Entropy')
    ax.plot(epochs, CROSS_test, 'r', label='Test Cross-Entropy')
    plt.title('Cross-Entropy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Value')
    plt.legend()
    ax.grid(True)
    fig.savefig("./CETest.png")


    epochs = range(1, len(ACC_train) + 1)
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.plot(epochs, ACC_train, 'b', label='Train Accuracy')
    ax.plot(epochs, ACC_test, 'r', label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Value')
    plt.legend()
    ax.grid(True)
    fig.savefig("./ACCTest.png")


    # Último teste
    mse = nn.CrossEntropyLoss()

    test_outputs = model(x_test)
    test_loss = mse(test_outputs, y_test).item()

    acc = (torch.argmax(test_outputs, 1) == torch.argmax(y_test, 1)).float().mean()

    print("Erro(CROSS) do teste: ", test_loss)
    print("Acurácia do teste: ", acc.item())


def testando():
    print("Olá")