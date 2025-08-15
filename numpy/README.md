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

### Descriptive Numpy FeedForward Functions and Performance Results

| Component / Metric         | Description                                                       | Equation (LaTeX)                                                | Value / Example Output                  |
|----------------------------|-------------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------|
| **Sequential Layer**       | Linear stack of layers built using NumPy                          | -                                                                | Implemented in NumPy                      |
| **Dense Layer**             | Fully connected layer where each neuron receives input from all neurons in previous layer | $y = f(Wx + b)$                                                  | Implemented in NumPy                      |
| **Dropout Layer**           | Randomly sets a fraction of inputs to 0 during training to prevent overfitting | -                                                                | Applied dropout for regularization        |
| **Activation (Sigmoid)**    | Maps input to range (0,1) for binary classification               | $\sigma(x) = \frac{1}{1 + e^{-x}}$                              | Used in output layer                      |
| **Activation (ReLU)**       | Rectified Linear Unit; sets negative values to zero               | $\text{ReLU}(x) = \max(0, x)$                                   | Used in hidden layers                     |
| **Binary Cross-Entropy Loss** | Measures error for binary classification tasks                   | $L = -\frac{1}{N} \sum_{i=1}^N \big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\big]$ | **0.4605**                                |
| **Accuracy**                | Proportion of correct predictions                                 | $\text{Accuracy} = \frac{\text{True}}{\text{Total}}$             | **98.0%**                                 |
| **Precision**               | Proportion of positive predictions that are correct               | $\text{Precision} = \frac{\text{True Positive}}{\text{All Positive}}$ | **98.0%**                                 |
| **Batch Size**              | Number of samples processed in one pass                           | -                                                                | **1500** samples (5 features each)        |
| **Execution Time**          | Time taken to run one batch                                       | -                                                                | **14.53 seconds**                          |


### Images
Below are the
<img width="1920" height="967" alt="Model_Params" src="https://github.com/user-attachments/assets/a8481a5e-9837-4fa0-972e-27f03c00ec58" />

<img width="1920" height="967" alt="Accuracy_and_Precision" src="https://github.com/user-attachments/assets/290ebdb2-3dbf-4006-895d-0fbd0ebff698" />

## Statistical Analysis Engine
Furthermore, we created a statistical Engine that runs all numpy based functions with greater accuracy and neat strcuture and can be used for a variety of tasks.

Here, we implemented, the following table

### Statistics Class Methods

| Method Name                 | Description                                     | Input                     | Output                   |
|------------------------------|-----------------------------------------------|---------------------------|--------------------------|
| `correlation`               | Compute correlation between datasets          | 2 arrays/lists           | Correlation coefficient  |
| `covariance`                | Compute covariance                            | 2 arrays/lists           | Covariance value         |
| `mean_numpy`                | Calculate mean using NumPy                     | 1 array/list             | Mean value               |
| `median_numpy`              | Calculate median using NumPy                   | 1 array/list             | Median value             |
| `min_max_and_range_numpy`   | Find min, max, and range using NumPy           | 1 array/list             | Tuple `(min, max, range)`|
| `min_max_scaling`           | Perform min-max normalization                  | 1 array/list             | Scaled array             |
| `mode_numpy`                | Compute mode using NumPy                       | 1 array/list             | Mode value               |
| `percentile`                | Calculate percentile values                    | 1 array/list + percentile| Percentile value         |
| `quartile`                  | Compute quartiles (Q1, Q2, Q3)                | 1 array/list             | Tuple `(Q1, Q2, Q3)`    |
| `skewness_and_kurtosis`     | Compute skewness and kurtosis                  | 1 array/list             | Tuple `(skewness, kurtosis)`|
| `standard_dev`              | Calculate standard deviation                   | 1 array/list             | Standard deviation       |
| `variance`                  | Calculate variance                             | 1 array/list             | Variance value           |
| `z_score_normalization`     | Perform z-score normalization                  | 1 array/list             | Normalized array         |

### Results
After a set of Arrays are passed, our functions and class modules calculated the following results:
#### Dataset Overview
[[ 0  1  4  3  8  5 12  7 16  9 20 11 24 13 28 15 32 17 36 19]
 [ 2  2  6  4 10  6 14  8 18 10 22 12 26 14 30 16 34 18 38 20]
 [ 4  3  8  5 12  7 16  9 20 11 24 13 28 15 32 17 36 19 40 21]
 ...
 [38 20 42 22 46 24 50 26 54 28 58 30 62 32 66 34 70 36 74 38]]

#### Descriptive Statistic Functions
## 📊 Descriptive Statistics Functions

| Function Name              | Description                                               | Equation (LaTeX)                                                |
|----------------------------|-----------------------------------------------------------|------------------------------------------------------------------|
| `mean`                     | Average of all values                                     | $\displaystyle \bar{x} = \frac{\sum_{i=1}^n x_i}{n}$            |
| `median`                   | Middle value when dataset is sorted                       | -                                                                |
| `mode`                     | Most frequent value(s)                                    | -                                                                |
| `minimum`                  | Smallest value in dataset                                 | $\displaystyle \min(x)$                                         |
| `maximum`                  | Largest value in dataset                                  | $\displaystyle \max(x)$                                         |
| `range`                    | Difference between max and min                            | $\displaystyle R = \max(x) - \min(x)$                           |
| `variance`                 | Spread of data around mean                                | $\displaystyle \sigma^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n}$ |
| `standard_deviation`       | Square root of variance                                   | $\displaystyle \sigma = \sqrt{\frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n}}$ |
| `covariance`               | Joint variability of two variables                        | $\displaystyle \text{Cov}(X,Y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n}$ |
| `correlation`              | Strength of linear relationship between two variables     | $\displaystyle r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$   |
| `z_score_normalization`    | Normalize to mean 0 and std. deviation 1                   | $\displaystyle z_i = \frac{x_i - \bar{x}}{\sigma}$              |
| `min_max_scaling`          | Scale between 0 and 1                                     | $\displaystyle x' = \frac{x - \min(x)}{\max(x) - \min(x)}$      |
| `percentile`               | Value below which a given percentage falls                | $P_k = \frac{k(n+1)}{100}$ (position formula)                   |
| `quartile`                 | 25%, 50%, and 75% cut points                              | $Q_1, Q_2, Q_3$ from sorted data                                |
| `skewness`                 | Asymmetry of distribution                                 | $\displaystyle \frac{\sum_{i=1}^n (x_i - \bar{x})^3}{n\sigma^3}$ |
| `kurtosis`                 | Peakedness of distribution                                | $\displaystyle \frac{\sum_{i=1}^n (x_i - \bar{x})^4}{n\sigma^4}$ |



