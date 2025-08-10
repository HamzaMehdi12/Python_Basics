import numpy as np
import time
import random
import matplotlib.pyplot as plt


from timeit import default_timer as timer
from numpy import random
from tqdm import tqdm

class Sequential():
    def __init__(self, *layers):
        "Creating the functionality of a Sequential class like a model class"
        self.layers = list(layers)
    
    def forward(self, x):
        self.x = x

        for layers in self.layers:
            self.x = layers.forward(self.x)
            return x
        
    def __repr__(self):
        layer_strings = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]

        return f"Sequential(\n" + "\n".join(layer_strings) + "\n)"
    
    def add_layers(self, layer):
        self.layers.append(layer)

    def summary(self):
        print("=" * 60)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 60)
        
        current_shape = None
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                if current_shape is None:
                    current_shape = (layer.input_size, "batch_size")
                
                output_shape = (layer.out_size, "batch_size")
                params = layer.W.size + layer.b.size
                total_params += params
                
                print(f"{layer.__class__.__name__} ({i})"[:24].ljust(25), end="")
                print(f"{str(output_shape)}"[:19].ljust(20), end="")
                print(f"{params}")
                
                current_shape = output_shape
            else:
                print(f"{layer.__class__.__name__} ({i})"[:24].ljust(25), end="")
                print(f"{str(current_shape) if current_shape else 'Same'}"[:19].ljust(20), end="")
                print("0")
        
        print("=" * 60)
        print(f"Total params: {total_params}")
        print("=" * 60)

class Dense():
    def __init__(self, in_, out_, activation=None):
        "Creates a dense layer like structure for the network. Here we implement Mulitplication and broadcasting"
        self.input_size = in_
        self.out_size = out_
        self.activation = activation

        #Setting weight and bias
        self.W = np.random.randn(self.input_size, self.out_size) * np.sqrt(2.0/self.input_size) # using He Initialization or Kaiming Initialization
        self.b = np.zeros((self.out_size, 1))

    def __call__(self, x):
        return self.forward(x) #call to make forward run automatically
    

    def forward(self, x):
        "Initializing the multiplication, broadcasting and vectorization"
        self.x = x

        self.Z = np.transpose(self.W) @ self.x + self.b #Neural activation linear pass (mx+ c)

        if self.activation == 'relu':
            return np.maximum(0, self.Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-self.Z))
        elif self.activation == 'tanh':
            return np.tanh(self.Z)
        else:
            print("Error, no activation, going with relu")
            return np.maximum(0, self.Z)
    
    def __repr__(self):
        "Showing class params"
        return f"Dense({self.input_size} -> {self.out_size}, activation={self.activation})"


class Dropout():
    def __init__(self, p= 0.25):
        "Creating dropout scenario in the model"
        self.p = p
        self.training = True
    def __call__(self, x):
        "Calls the functions forward automatically"
        self.x = x
        return self.forward(self.x)
    
    def forward(self, x):
        "Forward method for dropout layers"
        if self.training:
            mask = np.random.binomial(1, 1-self.p, x.shape) / (1-self.p) #inverted dropout
            return x * mask
        return x
    def __repr__(self):
        return f"Dropout(p = {self.p})"


class Forward_pass_numpy():
    def __init__(self, A, seed):
        "Initializing the forward pass for matrix operations"
        self.seed = seed
        self.A = A
        #lets begin
        random.seed(self.seed)
        print("Beginning the flow!")
        time.sleep(1)
        self.A_norm = self.nn_forward_pass(self.A, self.seed)


    def normalization(self, A, seed):
        "Normalizes the features attained"
        self.seed = seed
        self.A = A
        #Normalizing the input using mean and std_dev
        random.seed(self.seed)
        self.mean = np.mean(self.A, axis=1, keepdims=True)
        self.std_dev = np.std(self.A, axis=1, keepdims=True)

        self.A_norm = (self.A - self.mean) / self.std_dev
        print(f"Normalized array is now: \n{self.A_norm}")

        return self.A_norm
    
    def create_labels(self, A):
        "Creating labels of dataset in our model"
        #using multiple features
        self.A = A
        self.feat_sum = np.sum(self.A[:5, :], axis = 0) #Sum of all 5 features

        #Adding some noise
        self.noise = np.random.normal(0, 0.1, self.A.shape[1]) #shape[1] gives number of rows, shape[0] gives columns
        self.decision_score = self.feat_sum + self.noise

        #Converting to Binary labels
        self.labels = (self.decision_score > np.median(self.decision_score)).astype(int)
        return self.labels



    def loss_fn(self, y_true, y_pred):
        "Using loss function to cater loss and prediciton vs real value"
        self.y_true = y_true
        self.y_pred = y_pred
        self.bce = -np.mean((self.y_true * np.log(np.where(self.y_pred == 0, 1e-10, self.y_pred))) + ((1 - self.y_true)*np.log(np.where(1 - self.y_pred == 0, 1e-10, 1 - self.y_pred))))
        "Error solved using where, could also have used np.maximum(self.y_pred, 1e-10)"
        return self.bce

    def nn_forward_pass(self, A, seed):
        """This is the forward pass for the function to be processed
        What will we do!?
        Operations:
            - Mat Multiplications
            - Broadcasting
            - Vectorization
        Input:
            - Batch of arrays (1D batch with n features and x rows, shapr would be displayed)
        Output:
            - Prediction matrix of the type with predicitions of each feature vector in the model
                (Value would be 1 or 0, depending whether the salary is sufficient based on age, sex, etc)
        """
        #lets start
        self.seed = seed
        self.A = A
        random.seed(self.seed)
        print(f"Batch stacked and is as follows: \n{self.A}, \nsize: {len(self.A), len(self.A[0])}, \nshape: {self.A.shape}")
        time.sleep(3)
        print("Normalizing the array")
        self.A_norm = self.normalization(self.A, self.seed)
        self.A_norm[np.isnan(self.A_norm)] = 0
        time.sleep(3)
        #Calling the layers
        model = Sequential(
            Dense(5, 64, activation='relu'),
            Dropout(0.2),
            Dense(64, 128, activation='relu'),
            Dropout(0.4),
            Dense(128, 64, activation='relu'),
            Dropout(0.4),
            Dense(64, 32, activation='relu'),
            Dropout(0.3),
            Dense(32, 16, activation='relu'),
            Dropout(0.3),
            Dense(16, 2, activation='sigmoid')#Final layer
        )

        #Checking out our model and architecture
        print("Network Architecture")
        print("\n")
        #print("\n")
        #print("\n")
        print(model)
        print()
        model.summary()
        #creating true labels
        time.sleep(3)
        self.y_true = self.create_labels(self.A_norm)
        self.y_true[np.isnan(self.y_true)] = 0
        #print(f"True labels are as follows: \n{self.y_true}")
        #time.sleep(3)
        #Visualizing the data

        fig, axes = plt.subplots(2,2, figsize =(12,10))
        axes[0, 0].plot(self.A_norm, self.A_norm)
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.title('Normalized Data')

        time.sleep(1)
        print("Now creating the loop for the functionality to work and predict")
        #Now going for detections and predictions
        print("Running the model and making predictions!\n")
        self.y_pred = model.forward(self.A_norm)
        self.y_pred[np.isnan(self.y_pred)] = 0
        self.y_pred = self.create_labels(self.y_pred)
        time.sleep(1)

        #Calculating loss
        self.bce = self.loss_fn(self.y_true, self.y_pred)
        print(f"Loss in the function is: {self.bce}")

        #Already randomly initiated so no need for other steps
        #Now plotting
        axes[0, 1].scatter(self.y_true, self.y_pred)
        plt.xlabel('True vs Predicted values')
        plt.ylabel('Binary')
        plt.title('Results')

        #Printing loss
        axes[1, 0].bar(['Loss'], [self.bce])
        plt.xlabel('Loss')
        plt.ylabel('Loss')
        plt.title('Loss value')

        axes[1,1].plot()
        plt.tight_layout()
        plt.show()

            



    
    
if __name__ == "__main__":
    #Creating features for our nn
    print("Execution started!")
    random.seed(42)
    start = timer()
    feat_1 = np.random.normal(175, 10, 1500) #Height in cm
    feat_2 = np.random.normal(50, 10, 1500) #Weight in KG
    feat_3 = np.random.randint(18, 60, 1500)#Age 18+
    feat_4 = np.random.choice([0, 1], 1500)#Gender
    feat_5 = np.random.exponential(50000, 1500) #salary

    #Stacking in a vectorized stack row wise
    A = np.vstack([feat_1, feat_2, feat_3, feat_4, feat_5])
    #Passing to the function
    flow = Forward_pass_numpy(A, seed = 42)

    end = timer()

    total = end-start

    print(f"Function halted and took {total:.2f} seconds to complete.")
