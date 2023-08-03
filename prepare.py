import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def read_data():
    
    filename = 'icr-identify-age-related-conditions/train.csv'
    
    return pd.read_csv(filename)


def rename_columns(df):
                          
    df.columns = [col.lower() for col in df.columns]
    
    df = df.rename(columns = {'class': 'target'})
                   
    return df
    

def train_val_test(df, strat = 'target', seed = 42):
    
    train, val_test = train_test_split(df, train_size = 0.7, stratify = df[strat],
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5, stratify = val_test[strat],
                                 random_state = seed)
    
    return train, val, test


def fill_nulls(df, train):
    
    bq_neg_median = train[train.target == 0].bq.median()
    
    df.bq = df.bq.fillna(bq_neg_median)
    
    for col in df.columns[df.isna().sum() > 0]:
        
        df[col] = df[col].fillna(train[col].median())
    
    return df


def create_dummies(df):
    
    return pd.get_dummies(df, columns = ['ej'], drop_first = True)


def scale_data(train, val, test):
    
    scaler = MinMaxScaler()
    
    train.iloc[:, 1:-2] = scaler.fit_transform(train.iloc[:, 1:-2])
    
    val.iloc[:, 1:-2] = scaler.transform(val.iloc[:, 1:-2])
    
    test.iloc[:, 1:-2] = scaler.transform(test.iloc[:, 1:-2])
    
    return train, val, test


def make_x_and_y(train, val, test):
    
    X_train = train.drop(columns = ['target', 'id'])
    
    y_train = train.target
    
    X_val = val.drop(columns = ['target', 'id'])
    
    y_val = val.target
    
    X_test = test.drop(columns = ['target', 'id'])
    
    y_test = test.target
    
    return X_train, y_train, X_val, y_val, X_test, y_test