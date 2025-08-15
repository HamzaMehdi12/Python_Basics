Here we implement the basics of Numpy in Machine Learning, Deep Learning and AI.

## Forward-Pass using Numpy
We design a state of the art forward-pass, mimicing the functionality of a neural network, implemented using numpy library only...

We implement Matrix Multiplication, Matrix Vectorization and Matrix Broadcasting, while keeping only the Numpy library as the source of our model. Firstly the complete array of 1000+ batches was normalized using a novel normalization function. After nromaliozation, the model was sent to the architecture.

### Model Architecture.
Using this approach, we were able to design the following Architecture. The model depicts a linear regression Neural Network for out problem and generally performs great.
Network Architecture
Sequential(
  (0): Dense(5 -> 64, activation=relu)
  (1): Dropout(p = 0.2)
  (2): Dense(64 -> 128, activation=relu)
  (3): Dropout(p = 0.4)
  (4): Dense(128 -> 64, activation=relu)
  (5): Dropout(p = 0.4)
  (6): Dense(64 -> 32, activation=relu)
  (7): Dropout(p = 0.3)
  (8): Dense(32 -> 16, activation=relu)
  (9): Dropout(p = 0.3)
  (10): Dense(16 -> 2, activation=sigmoid)
)

<p align="center>

| Layer (type)  | Output Shape         | Params |
|:--------------|:--------------------:|-------:|
| Dense (0)     | (64, 'batch_size')    |   384  |
| Dropout (1)   | (64, 'batch_size')    |     0  |
| Dense (2)     | (128, 'batch_size')   |  8320  |
| Dropout (3)   | (128, 'batch_size')   |     0  |
| Dense (4)     | (64, 'batch_size')    |  8256  |
| Dropout (5)   | (64, 'batch_size')    |     0  |
| Dense (6)     | (32, 'batch_size')    |  2080  |
| Dropout (7)   | (32, 'batch_size')    |     0  |
| Dense (8)     | (16, 'batch_size')    |   528  |
| Dropout (9)   | (16, 'batch_size')    |     0  |
| Dense (10)    | (2, 'batch_size')     |    34  |

**Total params:** 19,602
</p>

The Sequential, Dense and Dropout layers, including the activation functions in Dense were coded using Numpy.

### Loss Function
To evaluate our model, a loss function was also created, from the Binary Cross_Entropy. 
***Loss in the function is: 0.46051701859880917***
### Function timeline
The complete file took 14.53 seconds to completely run a batch of ***1500***, with each row of 5 features and to produce an accuracy of ***98.0%*** and a Precision of ***98.0%***

The accuracy is calculated based on the following formula
### Accuracy = True / Total
Similarly, Precision is calculated as follows: 
### Precision = True Positive / All Positive

### Images
Below are the
<img width="1920" height="967" alt="Model_Params" src="https://github.com/user-attachments/assets/a8481a5e-9837-4fa0-972e-27f03c00ec58" />

<img width="1920" height="967" alt="Accuracy_and_Precision" src="https://github.com/user-attachments/assets/290ebdb2-3dbf-4006-895d-0fbd0ebff698" />


