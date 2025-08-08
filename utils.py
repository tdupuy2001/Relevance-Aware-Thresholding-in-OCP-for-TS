from math import exp
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm.autonotebook import tqdm
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
import matplotlib.pyplot as plt


#the indexation of the dataset starts at 1 then if you want to start the training from the beginning put start_train=1
#then if end_train=start_test-1 there is no step between train and test
#start_test-end_train=ahead
def train(X, Y, start_train, end_train, start_test, end_test, basemodel, **kwargs):
    # Initialization
    assert start_train>0, 'indexation of the dataset starts at 1'

    n = len(Y)
    test_size = end_test - start_test + 1
    train_size = end_train-start_train + 1
    assert test_size+train_size<=n

    y_pred = np.empty(test_size)
    y_test = np.empty(test_size)

    assert basemodel in ['RF','LR'], 'basemodel must be RF or LR.'
    if basemodel == 'RF':
        try:
            n_estimators = kwargs['n_estimators']
            min_samples_leaf = kwargs['min_samples_leaf']
            max_features = kwargs['max_features']
        except:
            raise ValueError("Arguments n_estimators, min_samples_leaf and max_features must be passed")
    print('Forecasting...')
    for t in tqdm(range(test_size)):
        x_train_t = np.transpose(X)[start_train-1+t:(end_train+t),]
        x_test_t = np.transpose(X)[(start_test+t),].reshape(1, -1)
        y_train_t = Y[start_train-1+t:(end_train+t)]
        y_test_t = Y[(start_test-1+t)]
        if basemodel == 'RF':
            reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                        random_state=1)
        elif basemodel == 'LR':
            reg = LinearRegression()
        reg.fit(x_train_t, y_train_t)
        y_pred_t = reg.predict(x_test_t)
        y_test[t]=y_test_t
        y_pred[t]=y_pred_t
    print('Forecasting finished') 
    return y_test, y_pred

def train_withoutX(Y, start_train, end_train, start_test, end_test, basemodel, sliding,**kwargs):
    # Initialization
    assert start_train>0, 'indexation of the dataset starts at 1'

    ahead = start_test-end_train
    assert ahead>=1, "test must be after train" 

    n = len(Y)
    test_size = end_test - start_test + 1
    assert test_size+end_train - start_train + 1<=n

    y_pred = np.empty(test_size)
    y_test = np.empty(test_size)

    assert basemodel in ['AR','Theta'], 'basemodel must be AR or Theta'
    if basemodel=='Theta':
        period_regressor=kwargs['period_regressor']
    print('Forecasting...')
    for t in tqdm(range(test_size)):
        if sliding==True:
            y_train_t = Y[start_train-1+t:(end_train+t)] 
            train_size = end_train - start_train + 1
        else:
            y_train_t = Y[start_train-1:(end_train+t)]
            train_size = end_train + t - start_train + 1
        y_test_t = Y[(start_test-1+t)]
        if basemodel == 'AR':
            model = AutoReg(y_train_t,lags=3).fit()
            y_pred_t = model.predict(start=train_size-1+ahead,end=train_size-1+ahead) #train_size-1 because according to the documentation AutoReg uses a 0-indexation
            y_pred_t = y_pred_t[0]
        # elif basemodel == 'Prophet':
        #     model = Prophet()
        #     model.fit(y_train_t)
        #     y_pred_t = model.predict(start=train_size-1+ahead,end=train_size-1+ahead) 
        #     y_pred_t = y_pred_t[0]
        elif basemodel == 'Theta':
            model = ThetaModel(y_train_t,period=period_regressor).fit()
            y_pred_t = model.forecast(theta=2).iloc[0]
        y_test[t]=y_test_t
        y_pred[t]=y_pred_t
    print('Forecasting finished') 
    
    return y_test, y_pred


def load_dataset(name):
    if name == "daily-climate":
        df = pd.read_csv('./datasets/daily-climate.csv')
        df.rename({'date': 'timestamp', 'meantemp': 'Y'}, axis=1, inplace=True)
        df.drop("Unnamed: 0", axis=1,inplace=True)
        df.set_index('timestamp', inplace=True)
    if name == "GOOGL":
        df = pd.read_csv('./datasets/djia.csv')
        df.rename({'Date': 'timestamp', 'Open': 'Y'}, axis=1, inplace=True)
        df=df[df['Name']=='GOOGL']
        df.drop('Name', axis=1, inplace=True)
        df.set_index('timestamp', inplace=True)
    if name == "AMZN":
        df = pd.read_csv('./datasets/djia.csv')
        df.rename({'Date': 'timestamp', 'Open': 'Y'}, axis=1, inplace=True)
        df=df[df['Name']=='AMZN']
        df.drop('Name', axis=1, inplace=True)
        df.set_index('timestamp', inplace=True)
    if name == "MSFT":
        df = pd.read_csv('./datasets/djia.csv')
        df.rename({'Date': 'timestamp', 'Open': 'Y'}, axis=1, inplace=True)
        df=df[df['Name']=='MSFT']
        df.drop('Name', axis=1, inplace=True)
        df.set_index('timestamp', inplace=True)
    df['Y'] = df['Y'].astype(float)
    df.interpolate(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[[col for col in df.columns if col != 'Y'] + ['Y']]
    return df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#mean as an argument to be general in the file "methods" (our function needs it)
def dev_sigmoid_c(c):
    def dev_sigmoid(s,q, mean):
        x=s-q
        dev_sig=c*sigmoid(c*x)*(1-sigmoid(c*x)) #use of form sig(1-sig) for numeric computation
        return dev_sig
    return dev_sigmoid

def f_w_v(w,v,alpha):
    def f(s, q, mean):
        x=s-q
        a=v/mean
        b=np.array([-np.log((1-alpha)/alpha) for _ in range(w.shape[0])])
        return np.sum(w[:, None] * sigmoid(a[:, None] * x + b[:, None]), axis=0)    
    return f

def dev_f_w_v(w,v,alpha):
    def dev_sum(s, q, mean):
        x=s-q
        a=v/mean
        b=np.array([-np.log((1-alpha)/alpha) for _ in range(w.shape[0])])
        z=a[:, None] * x + b[:, None]
        return np.sum(w[:, None] * a[:, None] * sigmoid(z) * (1-sigmoid(z)), axis=0) #use of form sig(1-sig) for numeric computation
    return dev_sum
