Here we implement the basics of Numpy in Machine Learning, Deep Learning and AI.

## Forward-Pass using Numpy
We design a state of the art forward-pass, mimicing the functionality of a neural network, implemented using numpy library only...

We implement Matrix Multiplication, Matrix Vectorization and Matrix Broadcasting, while keeping only the Numpy library as the source of our model.

### Model Architecture.
Using this approach, we were able to design the following Architecture
Network Architecture
Sequential(
  |(0): Dense(5 -> 64, activation=relu)|
  |(1): Dropout(p = 0.2)|
  |(2): Dense(64 -> 128, activation=relu)|
  (3): Dropout(p = 0.4)
  (4): Dense(128 -> 64, activation=relu)
  (5): Dropout(p = 0.4)
  (6): Dense(64 -> 32, activation=relu)
  (7): Dropout(p = 0.3)
  (8): Dense(32 -> 16, activation=relu)
  (9): Dropout(p = 0.3)
  (10): Dense(16 -> 2, activation=sigmoid)
)

------------------------------------------------------------
Layer (type)              Output Shape         Params

------------------------------------------------------------
Dense (0)                (64, 'batch_size')  384
Dropout (1)              (64, 'batch_size')  0
Dense (2)                (128, 'batch_size') 8320
Dropout (3)              (128, 'batch_size') 0
Dense (4)                (64, 'batch_size')  8256
Dropout (5)              (64, 'batch_size')  0
Dense (6)                (32, 'batch_size')  2080
Dropout (7)              (32, 'batch_size')  0
Dense (8)                (16, 'batch_size')  528
Dropout (9)              (16, 'batch_size')  0
Dense (10)               (2, 'batch_size')   34

----------------------------------------------------------
Total params: 19602



