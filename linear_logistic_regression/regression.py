#Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import time
#from tqdm import tqdm
from matplotlib.animation import FuncAnimation

#Logistic Regression
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Linear_Regression:
    def __init__(self):
        "This is the initialization of the class Regression for linear and logistic"
        self.params = {}
        self.derivatives = {}
        #self.train(lr, iter)
    
    def forward(self, X):
        "The forward method for linear regression"
        np.random.seed(42)
        self.X_f = X
        self.m = self.params['m']
        self.b = self.params['b']
        self.y_f_reg = np.multiply(self.m, self.X_f) + self.b
        return self.y_f_reg
    
    def cost_func(self, y, y_reg):
        "Cost functions of the predicted vs the actual dataset"
        self.y_c = y
        self.y_c_reg = y_reg
        cost = np.mean(np.power(self.y_c_reg - self.y_c, 2)) #calculates the cost of the difference between the predicted and the real values
        return cost
    
    def backprop(self, X, y, y_reg):
        "Back propagation of the values for error calculations"
        self.X_b = X
        self.y_b_reg = y_reg
        self.y_b = y
        df = (self.y_b_reg - self.y_b)
        dm = 2 * np.mean(np.multiply(self.X_b, df))
        db = 2 * np.mean(df)
        self.derivatives['dm'] = dm
        self.derivatives['db'] = db
    
    def update_params(self, lr):
        "Update parameters after the derivatives"
        self.params['m'] = self.params['m'] - lr * self.derivatives['dm']
        self.params['b'] = self.params['b'] - lr * 10 * self.derivatives['db']

    def performance(self, y_true, y_reg):
        "Calculating accuracy, precision, recall and F1 score"
        self.y_reg_per = y_reg
        self.y_true_per = y_true
        mse = np.mean(np.power(self.y_reg_per - self.y_true_per, 2))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.y_reg_per - self.y_true_per))
        ss_res = np.sum((self.y_true_per - self.y_reg_per)**2) 
        ss_tot = np.sum((self.y_true_per - np.mean(self.y_true_per))**2)
        
        r2 = 1 - (ss_res / ss_tot) 

        return mse, rmse, mae, r2


    def train(self, lr, iter):
        "training the function"
        np.random.seed(42)
        n_samples = 1000

        self.X = np.random.rand(n_samples, 1) * 10 #Shape (1000, 1), range (1, 10)
        self.X = self.X.flatten()
        split_index = int(0.6 * self.X.shape[0])
        self.X_train = self.X[:split_index].flatten()
        self.X_test = self.X[split_index: ].flatten()
        print(f"Shape of X_TRAIN: {self.X_train.shape}, X_test: {self.X_test.shape}")
        slope = 3
        b = 5
        #Noise = np.random.randn(n_samples, 1) * 2 #Gaussian noise

        self.y_true = slope * self.X + b
        self.y_true = self.y_true.flatten()
        split_index_y = int(0.6 * self.y_true.shape[0])
        self.y_true_train = self.y_true[:split_index_y].flatten()
        self.y_true_test = self.y_true[split_index_y: ].flatten()
        print(f"Shape of y_train: {self.y_true_train.shape}, y_test: {self.y_true_test.shape}")

        self.params['m'] = np.random.uniform()
        self.params['b'] = np.random.uniform()
        print(f"Diff of size: {int(self.X.shape[0]) - split_index}")

        self.loss = []
        self.test_loss = []
        #plotting the data and the initial curve of predictions
        fig, ax = plt.subplots()
        #x_vals = np.linspace(min(self.X_train), max(self.X_train), split_index).flatten()
        line_train, = ax.plot(self.X_train. flatten(), self.params['m'] * self.X_train.flatten() + self.params['b'], color = 'red', linewidth = 4, linestyle = '-', label = 'Regression Line')
        #x_vals_test = np.linspace(min(self.X_test), max(self.X_test), int(self.X.shape[0]) - split_index).flatten()
        line_test, = ax.plot(self.X_test.flatten(), self.params['m'] * self.X_test.flatten() + self.params['b'], color = 'blue', linewidth = 4, linestyle = '--', label = 'Regression_test_Line')
        ax.scatter(self.X, self.y_true, marker='o', color = 'green', label = 'Training Data')
        ax.set_ylim(0, max(self.y_true) + 1)

        self.mse = []
        self.mae = []
        self.rmse = []
        self.r2 = []                

        self.mse_test = []
        self.rmse_test = []
        self.mae_test = []
        self.r2_test = []


        def predict_linear(iters):
            "Predicting a linear function"
            self.y_pred_train = self.forward(self.X_train) #Forward progression
            self.cost_loss = self.cost_func(self.y_true_train, self.y_pred_train)#calculating loss
            self.backprop(self.X_train, self.y_true_train, self.y_pred_train)#back propagation
            self.update_params(lr) #Updating params
            #if iters % 10 == 0:
                #print(f"Paramters updated are m: {self.params['m']} and b: {self.params['b']}")
            line_train.set_data(self.X_train.flatten(), self.params['m'] * self.X_train.flatten() + self.params['b'])
            line_test.set_data(self.X_test.flatten(), self.params['m'] * self.X_test.flatten() + self.params['b'])
            self.loss.append(self.cost_loss) #Appending loss for future use
            mse, rmse, mae, r2 = self.performance(self.y_true_train, self.y_pred_train)
            self.mse.append(mse)
            self.rmse.append(rmse)
            self.mae.append(mae)
            self.r2.append(r2)
            #print(f"Y_train: {self.y_true_train} and prediction: {self.y_pred_train}")

            #Now for testing
            self.y_pred_test = self.forward(self.X_test)
            self.test_cost_loss = self.cost_func(self.y_true_test, self.y_pred_test)
            self.test_loss.append(self.test_cost_loss) #Appending for future
            mse_test, rmse_test, mae_test, r2_test = self.performance(self.y_true_test, self.y_pred_test)
            self.mse_test.append(mse_test)
            self.rmse_test.append(rmse_test)
            self.mae_test.append(mae_test)
            self.r2_test.append(r2_test)
            if iters % 10 == 0:
                print(f"Iterations: {iters + 1}") 
                print(f"Train -> Loss: {self.cost_loss}, Mean Squared Error: {mse: .2f}, Root Mean Squared Error: {rmse: .2f}, Mean Absolute Error: {mae: .2f}, R2: {r2: .2f}")
                print(f"Test -> Loss: {self.test_cost_loss}, Mean Squared Error: {mse_test: .2f}, Root Mean Squared Error: {rmse_test: .2f}, Mean Absolute Error: {mae_test: .2f}, R2: {r2_test: .2f}")
            return line_train, line_test

           
        ani = FuncAnimation(fig, predict_linear, frames=iter, interval=200, blit=True, repeat=True)
        #Predicting on Test dataset
        ani.save('Linear_Regression.gif', writer='pillow')

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

        iters = [x for x in range(0, len(self.mse), 1)]


        #Plotting the training and test errors
        plt.figure(figsize=(10,6))
        plt.plot(iters, self.mse, 'b', linewidth = 2, label='Training MSE', alpha=0.8)
        plt.plot(iters, self.mse_test, 'g', linewidth = 2, label='Test MSE', alpha=0.8)
        plt.plot(iters, self.rmse, 'b-', linewidth = 2, label='Training RMSE', alpha=0.8)
        plt.plot(iters, self.rmse_test, 'g-', linewidth = 2, label='Test RMSE', alpha=0.8)
        plt.plot(iters, self.mae, 'b--', linewidth = 2, label='Training MAE', alpha=0.8)
        plt.plot(iters, self.mae_test, 'g--', linewidth = 2, label='Test MAE', alpha=0.8)
        plt.plot(iters, self.r2, 'r-', linewidth = 2, label='Training R2', alpha=0.8)
        plt.plot(iters, self.r2_test, 'y-', linewidth = 2, label='Test R2', alpha=0.8)


        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Error Values', fontsize=12)
        plt.title('Multiple Error Metrics Over Training and Testing', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


        iters = [x for x in range(0, len(self.loss), 1)]
        iters_test = [x for x in range(0, len(self.test_loss), 1)]

        plt.figure(figsize=(10,6))
        plt.plot(iters, self.loss, 'b', linewidth = 2, label='Training MSE', alpha=0.8)
        plt.plot(iters_test, self.test_loss, 'g', linewidth = 2, label='Test MSE', alpha=0.8)

        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss Values', fontsize=12)
        plt.title('Loss for Training and Testing', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


        return self.params, self.loss


class Logistic_Regression:
    def __init__(self):
        "Implementing the logistic regression"
        self.params = {}
        self.derivatives = {}
        self.loss = []
        self.acc = []
        self.prec = []

    def forward_pass(self, X):
        "Forward pass borrowed from Linear Regression"
        np.random.seed(42)
        self.X_f = X
        self.m = self.params['m']
        self.b = self.params['b']
        self.y_f_pred = np.dot(self.X_f, self.m) + self.b
        return self.y_f_pred

    def sigmoid(self, z):
        "Sigmoid function for Logistic Regression"
        return 1 / (1 + np.exp(-z)) 
    
    def cost_fn(self, y, y_pred):
        "Cost function, borrowed from Linear Regression"
        "Cost functions of the predicted vs the actual dataset"
        self.y_c = y
        self.y_c_pred = y_pred
        l = len(self.y_c)
        #cost = - (1/l) * np.sum(self.y_c*np.log(self.y_c_pred) + (1-self.y_c)*np.log(1-self.y_c_pred)) #calculates the cost of the difference between the predicted and the real values
        cost = np.mean(np.power(self.y_c_pred - self.y_c, 2))
        return cost
    
    def backprop(self, X, y, y_pred):
        "Back Propagation, also borrowed from Linear Regression"
        self.X_b = X
        self.y_b_pred = y_pred
        self.y_b = y
        df = (self.y_b_pred - self.y_b)
        dm = 2 * np.mean(np.dot(np.transpose(self.X_b), df))
        db = 2 * np.mean(df)
        self.derivatives['dm'] = dm
        self.derivatives['db'] = db
    
    def update_params(self, lr):
        "Updating params also borrowed from Linear Regression"
        self.params['m'] = self.params['m'] - lr  * 50 * self.derivatives['dm']
        self.params['b'] = self.params['b'] - lr * 50 * self.derivatives['db']
        
    def acc_prec(self, y, y_pred):
        "Accuracy and Prediction for the model"
        accuracy = np.mean(y_pred == y)
        TP = np.sum((y_pred == 1) & (y == 1))
        FP = np.sum((y_pred == 1) & (y == 0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        return accuracy, precision

    def fit(self, X, y, lr, iter):
        "Fitting the data"
        for i in range(iter):
            time.sleep(0.005)
            z = self.forward_pass(X)
            self.y_train_preds = self.sigmoid(z) #Forward pass
            self.cost_loss = self.cost_fn(y, self.y_train_preds) #Cost function
            self.backprop(X, y, self.y_train_preds) #back propagation
            self.update_params(lr)
            self.loss.append(self.cost_loss)
            #acc, prec = self.acc_prec(self.y_train, self.y_train_preds)
            #self.acc.append(acc)
            #self.prec.append(prec)

            if i%100 == 0:
                print(f"Train Results -> Iterations: {i}, Loss: {self.cost_loss: .4f}")
                #print(f"Params-> m: {self.params['m']}, b: {self.params['b']}")
        return self.y_train_preds
    
    def predict(self, X):
        "Predicting the model"
        return(self.sigmoid(self.forward_pass(X)) >= 0.5).astype(int)
        
    def train(self, lr, iters):
        "This is the training and fitting of the logistic regression"
        np.random.seed(42)
        self.X = np.random.randn(1000, 2) * 10
        self.y = (self.X[:, 0] + self.X[:, 1] > 10).astype(int)
        #Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        #Scaling the dataset
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        a, b = self.X_train.shape

        self.params['m'] = np.zeros(b)
        self.params['b'] = 0.0

        y_preds = self.fit(self.X_train, self.y_train, lr, iters)

        plt.scatter(self.X_train[:, 0], self.y_train, color="red", label = "Data")
        plt.scatter(self.X_train[:, 1], y_preds, color="blue", label = "Preds")
        plt.xlabel("X vals")
        plt.ylabel("Probability")
        plt.title("Performance of our Model")
        plt.legend()
        plt.show()

        time.sleep(2)

        plt.plot(self.loss)
        plt.title("Loss Curve for Training")
        plt.xlabel("Iterations")
        plt.ylabel('Cost')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #Now on Test data
        predictions = self.predict(self.X_test)
        self.acc_test, self.prec_test = self.acc_prec(self.y_test, predictions)
        plt.scatter(self.X_test[:, 0], self.y_test, color="red", label = "Data")
        plt.scatter(self.X_test[:, 1], predictions, color="blue", label = "Preds")
        plt.xlabel("X vals")
        plt.ylabel("Probability")
        plt.title("Performance of our Model on Test data")
        plt.legend()
        plt.show()

        print(f"Model Accuracy: {self.acc_test * 100: .2f}%, Model precision: {self.prec_test *100: .2f}%")
        #print(f"True test values: {self.y_test} \nvs\npredicted values: {predictions}")

        return self.loss
    
if __name__ == "__main__":
    print("Lets Start")
    print("Running model Linear Regression")
    model = Linear_Regression()
    params, loss = model.train(0.001, 100)
    #Now running Logistic Regression
    print("Running Model Logistic Regression")
    time.sleep(2)
    model1 = Logistic_Regression()
    loss = model1.train(0.002, 1000)

    method_list_linear = [method for method in dir(Linear_Regression) if method.startswith('__') is False]
    print(f"Linear Regression Mehtods: {method_list_linear}")
    method_list_logistic = [method for method in dir(Logistic_Regression) if method.startswith('__') is False]
    print(f"Logistic Regression Methods: {method_list_logistic}")

    print("Completed models")