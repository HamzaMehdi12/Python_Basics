import pandas as pd
import numpy as np
import time
from scipy import stats
from sklearn.preprocessing import StandardScaler

class LabelEncoding:
    def fit(self, series):
        "Custom label encoder as no other library is to be used"
        #Saving all unique, non-missing values
        self.classes_ = series.dropna().unique()

        #Building mapping: cat -> int
        self.mapping = {cat: idx for idx, cat in enumerate(self.classes_)}

        #Building reverse: int -> cat
        self.inverse_mapping = {idx: cat for cat, idx in self.mapping.items()}
        return self
    
    def transform(self, series):
        "Converts categories to numbers"
        return series.map(self.mapping)
    
    def fit_transform(self, series):
        "Fit and transform in one go"
        return self.fit(series).transform(series)
    
    def inverse_transform(self, series):
        "Converts numbers back to categorical"
        return series.map(self.inverse_mapping)
    
class Imputer:
    def __init__(self, strategy = "mean", fill_val = None):
        "Custom imputer for our Data"
        "Strategy:"
        "mean -> fill with mean of column"
        "median -> fill with median"
        "most_frequent -> fill with mod"
        "constant -> fill with the value you provide"

        self.strategy = strategy
        self.fill_val = fill_val
        self.val_ = {}

    def fit(self, df):
        "Fitting the values to be imputed"
        for col in df.columns:
            series = df[col]
            if self.strategy == "mean":
                self.val_[col] = series.mean()
            elif self.strategy == "median":
                self.val_[col] = series.median()
            elif self.strategy == "most_frequent":
                self.val_[col] = series.mod()[0]
            elif self.strategy == "constant":
                if self.fill_val is None:
                    raise ValueError("Fill value must be a valid object for constant strategy")
                self.val_[col] = self.fill_val
            else:
                raise ValueError("Invalid strategy: Choose between mean/median/most_frequent/constant")
        return self
    
    def transform(self, df):
        "Replacing Nan values with learned vals"
        for col, value in self.val_.items():
            df[col] = df[col].fillna(value)
        return df
    
    def fit_transform(self, df):
        "Fit and transform in one go"
        return self.fit(df).transform(df)

class Data_Clensing:
    def __init__(self, df):
        "This is the cleaning of the dataset"
        self.df = df
        with pd.option_context('display.max_columns', None):
            print(f"Data received is as follows:") 
            print(self.df.head(5).to_string())
        #Lets start data cleaning
        self.Data_Preprocessing()
        time.sleep(3)
        self.saving_data()

    def conv_m(self, X):
        "Convert values from digits to million"
        if isinstance(X, str):
            X = X.replace('$', '').strip()
            if X.endswith('M'):
                return float(X[:-1]) * 1_000_000
            try:
                return float(X)
            except:
                return None
        return X

    def Data_Preprocessing(self):
        "Cleaning the data for better visualization"
        print(f"Generating Data Info: \n")
        self.df.info(verbose=True) #Never works in print
        #Cheking Null/Nan values and adding 0 or average to them
        print("Checking Nan and Null values in numeric data")
        time.sleep(2)
        if self.df.isna().any().any(): #For checking the columns (first any) and for checking the whole dataframe
            print("Nan or Null values detected.")
            time.sleep(1)
            self.df = self.df.fillna(0)
            print("Completed filling nan values")
            time.sleep(1)
        else:
            print("No values found")
            time.sleep(1)
        
        #Now dropping the duplicates
        print("Dropping duplicates")
        time.sleep(2)
        if self.df.duplicated().any():
            print("Encountered duplicates")
            time.sleep(1)
            self.df = self.df.drop_duplicates()
            print("Dropped all duplicates")
            time.sleep(1)
        else:
            print("No duplicates encountered")
            time.sleep(1)

        #All duplicates have been dropped. Now 2 things, first amend the year and then, the votes
        #Cleaning the year colums
        print("Feature Engineering")
        time.sleep(2)
        self.df['YEAR'] = self.df['YEAR'].str.replace(r'[^0-9\-]', '', regex = True) #Regular expression to remove any non number digits that are not from 0 to 9, and do not have a -
        self.df[['start_year', 'end_year']] = self.df['YEAR'].str.extract(r'(?P<start>\d{4})?-?(?P<end>\d{4})?') #Another regular expression, it finds the 4 number for year before and after the dash. Each for start and end year
        self.df['start_year'] = self.df['start_year'].fillna(self.df['end_year']).astype(float)#Data has many entries that do not have an year (start or end). For those, we fill with what values we already have
        self.df['end_year'] = self.df['end_year'].fillna(self.df['start_year']).astype(float)#Same here

        #converting int ot datetime
        self.df['start_year'] = pd.to_datetime(self.df['start_year'], format='%Y', errors='coerce')#Converting to a datetime string
        self.df['end_year'] = pd.to_datetime(self.df['end_year'], format='%Y', errors='coerce')
        self.df = self.df.drop('YEAR', axis = 1)

        if self.df.isna().any().any():
            print("New Nan values after feature engineering")
            time.sleep(1)
            self.df.dropna(subset=['start_year', 'end_year']).reset_index(drop=True)
            print("Dropped Entire rows of nan values")
            time.sleep(1)
        else:
            print("No Nan Values to be dropped")
            time.sleep(1)

        #Now for the votes
        self.df['VOTES'] = self.df['VOTES'].str.replace(',', '').fillna(0).astype(int)
        #Working on outliers now
        #First, for Gross
        self.df['Gross'] = self.df['Gross'].apply(self.conv_m)

        non_zero_q = self.df[self.df['Gross']!= 0]['Gross']
        Q1 = non_zero_q.quantile(0.25)
        Q3 = non_zero_q.quantile(0.75)
        IQR = Q3 - Q1
        #print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")

        lower_bound = Q1 - (1.5 * IQR)

        median_val = non_zero_q.median()
        #print(f"Median is as follows: {median_val}")

        self.df['Gross'] = self.df['Gross'].apply(lambda x: median_val if x == 0 or x < lower_bound else x)

        self.df = self.df.drop(['ONE-LINE', 'STARS'], axis = 1)
        
        #Now converting our columns to numerical values for further preprocessing. We would use the Label Encoder
        print("Performing Label Encoding")
        time.sleep(2)
        cat_cols = self.df.select_dtypes(include=['object']).columns

        for cat in cat_cols:
            encoder = LabelEncoding()
            self.df[cat] = encoder.fit_transform(self.df[cat])

        print("Finished label encoding")
        time.sleep(1)

        print("Performing imputing")
        time.sleep(2)
        num_cols = self.df.select_dtypes(include = ['int32', 'int64', 'float64']).columns
        self.df_num = self.df[num_cols]

        self.df = self.df.drop(num_cols, axis = 1)

        imputer = Imputer(strategy="mean")

        self.df_num = imputer.fit_transform(self.df_num)

        print("Finished imputing")
        time.sleep(1)

        #Now going for standard scaling
        print("Scaling the dataset")
        time.sleep(2)

        scaler = StandardScaler()

        self.df_scaled = pd.DataFrame(scaler.fit_transform(self.df_num), columns=num_cols)

        self.df = pd.concat([self.df, self.df_scaled], axis = 1)
        print("Dataset has been scaled")
        time.sleep(1)

        #self.df.info()
        print(f"Now the dataset is as follows: \n")
        time.sleep(2)
        with pd.option_context('display.max_columns', None):
            print(self.df.head(5).to_string())#to_string for whole line output without line wrapping

        return self


    def saving_data(self):
        "After cleaning, saving the data on csv on device"
        print("Saving the file")
        time.sleep(2)
        self.df.to_csv('processed_data.csv', index=False)
        print("File saved!")
        time.sleep(1)
        return self

if __name__ == "__main__":
    df = pd.read_csv('movies.csv')
    Data = Data_Clensing(df)
    print("Completed the entire structure...... Program closing in 3 seconds")
    time.sleep(3)