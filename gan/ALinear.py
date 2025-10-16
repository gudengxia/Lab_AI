import torch
import torch.nn as nn
import random
import pandas as pd
import time
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), 0.03)
    
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
class Dataset:
    def __init__(self, train_size, test_szie):
        w = torch.tensor([2, -3.4])
        b = 4.2
        noise = 0.01 

        self.train_size = train_size
        self.test_size = test_szie

        n = train_size + test_szie
        self.x = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.x, w.reshape((-1, 1))) + b + noise
        #self.y = y.reshape([len(y)])
    
    def get_train_data(self):
        x_train = self.x[range(0, self.train_size)]
        y_train = self.y[range(0, self.train_size)]
        return (x_train, y_train)
    
    def get_test_data(self):
        x_test = self.x[range(self.train_size, -1)]
        y_test = self.y[range(self.train_size, -1)]
        return (x_test, y_test)
    
    def save(self, file):
        data = torch.cat((self.x, self.y), 1)
        pf = pd.DataFrame(data)
        pf.to_csv(file, index = False, header=["x0", "x1", "y"])


class Learner:
    def __init__(self, epochs, batch_size, lr = 0.0005):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, model, data):
        #data = Dataset(1024, 128)
        x_train, y_train = data.get_train_data()
        x_test, y_test = data.get_test_data()

        sgd = model.configure_optimizers()
        n_batch = len(x_train) // self.batch_size
        idxs = list(range(0, len(x_train))) 
        random.shuffle(idxs)

        total_time = 0
        for epoch in range(self.epochs):
            total_loss = 0.0

            start_time = time.perf_counter()
            for i in range(n_batch):
                x = x_train[range(i * self.batch_size, i * self.batch_size + self.batch_size)]
                y = y_train[range(i * self.batch_size, i * self.batch_size + self.batch_size)]
                #x = x.to_device()
                #y = y.to_device()

                y_hat = model.forward(x)
                loss = model.loss(y_hat, y)
                sgd.zero_grad()
                loss.backward()
                sgd.step()

                total_loss += loss
        
            end_time = time.perf_counter()
            diff = end_time - start_time
            total_time += diff
            avg_loss = total_loss / n_batch
            print("Epoch consumes ", diff)
            print("Epoch {} -- avg_loss: {}\n".format(epoch, avg_loss))
        
        print("Cosume time :", total_time )
        y_hat = model(torch.tensor([1.0, 1.0]))
        print(y_hat)

if __name__ == "__main__":
    d = Dataset(1024, 128)
    d.save("./../dataset/regression.csv")
    x_train, y_train = d.get_train_data()
    x_test, y_test = d.get_test_data()
    
    m = LinearRegressionModel(2, 1)

    leaner = Learner(10, 128) 
    leaner.fit(m, d)
