import numpy as np
import time

from scipy import stats as st
from scipy.stats import skew, kurtosis
from tqdm import tqdm

class Stats_Analysis_Engine:
    def __init__(self, A):
        "Class for statisitcal analysis engine"
        self.A = A 
        print(f"We have received an array \n{self.A} or rows: {len(self.A)} and cols: {len(self.A[0])}")
        time.sleep(2)

        print("Calculating mean")
        self.mean = self.mean_numpy(self.A)
        print("Calculating the variance")
        self.var = self.variance(self.A)
        print("Calculating the Standard Deviation")
        self.std_dev = self.standard_dev(self.A)
        print("Calculating the median of an array")
        self.A_median = self.median_numpy(self.A)
        print("Calulating the mode of an array")
        self.A_mode = self.mode_numpy(self.A)
        print("Calculating the minimum, maximum and range of array value of the Array")
        self.A_min, self.A_max, self.A_range = self.min_max_and_range_numpy(self.A)
        print("Calculating covariance of the array")
        self.A_cov = self.covariance(self.A)
        print("Calculating the correlation of Array with itself")
        self.A_corr = self.correlation(self.A)
        print("Calculating the z_score_normalization of the array from mean and standard deviation")
        self.A_z_scores = self.z_score_normalization(self.A, self.mean, self.std_dev)
        print("Calculating the Min_Max_Scaling of the array")
        self.A_min_max_scaler = self.min_max_scaling(self.A, self.A_min, self.A_max)
        print("Calculating the percentile of array at random percentages")
        self.A_perc = self.percentile(self.A)
        print("Calculating the quantile of array at 25, 50, 75 and 100%")
        self.A_quar = self.quartile(self.A)
        print("Finally, calculating skewness and kurtosis")
        self.A_skew, self.A_kurt = self.skewness_and_kurtosis(self.A)

        print("Completed the statistical library!")



    def mean_numpy(self, A):
        "Numpy calculation of mean of an array"
        self.B = A
        mean = np.mean(self.B)
        print(f"Mean of array is: {mean: .2f}")
        time.sleep(1)
        return mean 
    
    def variance(self, A):
        "Calculating the variance of array"
        self.B = A
        var = np.var(self.B)
        print(f"Variance of array is: {var: .2f}")
        time.sleep(1)
        return var
    
    def standard_dev(self, A):
        "Calculating the standard deviation of the array"
        self.B = A
        std_dev = np.std(self.B)
        print(f"Standard deviation of array is: {std_dev: .2f}")
        time.sleep(1)
        return std_dev
    
    def median_numpy(self, A):
        "Calculating the median of the array"
        self.B = A
        A_median = np.median(self.B)
        print(f"Median of array is: {A_median: .2f}")
        time.sleep(1)
        return A_median
    
    def mode_numpy(self, A):
        "Caluclating the mode of an array"
        self.B = A
        A_mode = st.mode(self.B)
        print(f"Mode of array is: {A_mode}")
        time.sleep(1)
        return A_mode
    
    def min_max_and_range_numpy(self, A):
        "Finding the minimum and maximum value in an array"
        self.B = A
        min = np.min(self.B)
        max = np.max(self.B)
        range = np.ptp(self.B, axis=1)
        print(f"Minimum: {min: .2f}, Maximum: {max: .2f} and range: {range} values found in the array")
        time.sleep(1)
        return min, max, range
    
    def covariance(self, A):
        "Measures the strength of correlation between set of variables"
        self.B = A
        cov = np.cov(self.B)
        print(f"Covariance of the array is: {cov}")
        time.sleep(1)
        
        return cov
    
    def correlation(self, A):
        "Correlation between 2 Arrays"
        self.B = A.flatten()
        self.C = np.copy(self.B)
        corr = np.correlate(self.B, self.C)
        print(f"Correltion between the arrays is as follows: {corr}")
        time.sleep(1)

        return corr
    
    def z_score_normalization(self, A, mean, std):
        "Calculates the normalization of the array"
        self.B = A
        self.B_mean = mean
        self.B_std = std

        z_score = (self.B - self.B_mean) / self.B_std

        print(f"Z_scores are as follows: {z_score}")
        time.sleep(1)

        return z_score
    
    def min_max_scaling(self, A, min, max):
        "Normalization of an arrya using min_max_scaling"
        self.B = A
        self.B_min = min
        self.B_max = max

        min_max = (self.B - self.B_min) / (self.B_max - self.B_min)
        print(f"MinMaxScaler of the array is as follows: {min_max}")
        time.sleep(1)

        return min_max
    
    def percentile(self, A):
        "Calulates the percentile or the element if the percentile is passed"
        self.B = A
        random = np.random.randint(0, 100)
        perc = np.percentile(self.B, random, axis=1, keepdims=True)

        print(f"Percentile for array at {random} percentage is: {perc}")
        time.sleep(1)

        return perc
    
    def quartile(self, A):
        "Calculating the quartile of array at 25% step"
        self.B = A
        self.quar = []

        j = 0

        for i in range(0, 101, 25):
            self.quar.append(np.quantile(self.B, i/100))
            print(f"Quartile at {i}% is: {self.quar[j]}")
            j += 1
            time.sleep(1)

        return self.quar 
    
    def skewness_and_kurtosis(self, A):
        "Skewness is the measure to shape the distriburion"
        "We also have Kurtosis, which measures the frequency of distribution"
        "We will implement both"

        self.B = A
        skewness = skew(self.B, axis=0, bias=True)
        kurt = kurtosis(self.B, axis=0, bias=True)
        
        print(f"The skew is: {skewness} and the kurtosis is: {kurt}")
        time.sleep(1)

        return skewness, kurt

if __name__ == "__main__":
    #start_time = time()

    print("Enter rows of Array: ")
    rows = int(input())
    print("Enter cols of Array: ")
    cols = int(input())

    A = []

    for i in tqdm(range(rows)):
        A.append([])
        time.sleep(0.15)
        for j in range(cols):
            if j%2 == 0:
                A[i].append(2 * (i+j))
            else:
                A[i].append(i + j) 
    A_numpy = np.array(A)
    Stats = Stats_Analysis_Engine(A_numpy)

    print("Class members are displayed below:")
    method_list = [method for method in dir(Stats_Analysis_Engine) if method.startswith('__') is False] #printing the methods present in the class
    print(method_list)

    print("Exiting the function in 3 seconds!")
    time.sleep(3)



    #end_time = time()
    #print(f"Total time for the execution: {end_time - start_time: .2f}")
    