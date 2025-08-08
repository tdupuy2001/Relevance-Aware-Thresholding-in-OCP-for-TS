import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tqdm.autonotebook import tqdm
from statsmodels.tsa.forecasting.theta import ThetaModel


def ECI(y_test, y_pred, alpha, q1, lr, f_dev,t_burnin=100):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    print('Starting ECI') 
    for t in tqdm(range(test_size)):
        t_mu = t
        t_mu_min = max(t_mu - t_burnin, 0)
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] < scores[t])
        err[t] = err_t 
        mean = np.abs(np.mean(scores[t_mu_min:t_mu]-q[t_mu_min:t_mu])) if t>0 else 1
        if t < test_size - 1:
            q[t+1] = max(q[t] + lr *(err_t-alpha+(scores[t]-q[t])*f_dev(scores[t],q[t],mean)),0)  
    print('End')
    return y_lowers, y_uppers, err

def ECI_full(y_test, y_pred, alpha, q1, lr, f, f_dev,t_burnin=100):
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    print('Starting ECI') 
    for t in tqdm(range(test_size)):
        t_mu = t
        t_mu_min = max(t_mu - t_burnin, 0)
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) # same as err=float(q[i] < scores[t])
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_mu_min:t]-q[t_mu_min:t]))) if t>0 else err_t 
        mean = np.abs(np.mean(scores[t_mu_min:t_mu]-q[t_mu_min:t_mu])) if t>0 else 1
        if t < test_size - 1:
            q[t+1] = max(q[t]+lr *(smooth_err[t]-alpha+(scores[t]-q[t])*f_dev(scores[t],q[t],mean)),0)  
    print('End')
    return y_lowers, y_uppers, err


def mytan(x):
    if x >= np.pi/2:
        return np.infty
    elif x <= -np.pi/2:
        return -np.infty
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI):
    if KI == 0:
        return 0
    tan_out = mytan(x * np.log(t)/(Csat * t))
    out = KI * tan_out
    return  out

def PID_log(y_test, y_pred, alpha, q1, lr, Csat, KI, period_scorecaster=5, t_burnin=100, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    print('Starting PID log')
    for t in tqdm(range(test_size)):
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        x = np.sum(err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next threshold
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1: model.forecast() forecasts the next time step
                q[t+1]=max(scorecasts[t]+integrator+lr*(err_t-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr*(err_t-alpha),0)
    print('End')
    return y_lowers, y_uppers, err

def PID_log_half_smooth(y_test, y_pred, alpha, q1, lr, Csat, KI, f, period_scorecaster=5, t_burnin=100, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    print('Starting PID log half smooth')
    for t in tqdm(range(test_size)):
        t_mu = t
        t_mu_min = max(t_mu - t_burnin, 0)
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t]#symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_mu_min:t]-q[t_mu_min:t]))) if t>0 else err_t
        x = np.sum(smooth_err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next threshold
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1: model.forecast() forecasts the next time step
                q[t+1]=max(scorecasts[t]+integrator+lr*(smooth_err[t]-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr*(err_t-alpha),0)
    print('End')
    return y_lowers, y_uppers, err, smooth_err

def PID_log_half_smooth_bis(y_test, y_pred, alpha, q1, lr, Csat, KI, f, period_scorecaster=5, t_burnin=100, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    print('Starting PID log half smooth bis')
    for t in tqdm(range(test_size)):
        t_mu = t
        t_mu_min = max(t_mu - t_burnin, 0)
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_mu_min:t]-q[t_mu_min:t]))) if t>0 else err_t
        # if np.mean(scores[t_mu_min:t]-q[t_mu_min:t])<0:
        #     smooth_err[t]=1-smooth_err[t]
        x = np.sum(err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next threshold
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1: model.forecast() forecasts the next time step
                q[t+1]=max(scorecasts[t]+integrator+lr*(smooth_err[t]-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr*(smooth_err[t]-alpha),0)
    print('End')
    return y_lowers, y_uppers, err, smooth_err

def PID_log_full_smooth(y_test, y_pred, alpha, q1, lr, Csat, KI, f, period_scorecaster=5, t_burnin=100, is_scorecast=False):
    # Initialization
    test_size = y_test.shape[0]
    y_lowers = np.empty(test_size,dtype=float)
    y_uppers = np.empty(test_size,dtype=float)
    q = np.full(test_size,q1,dtype=float)
    err = np.empty(test_size,dtype=float)
    smooth_err = np.empty(test_size,dtype=float)
    scores=np.abs(y_test-y_pred)
    scorecasts = np.empty(test_size,dtype=float)
    print('Starting PID log full smooth')
    for t in tqdm(range(test_size)):
        t_mu = t
        t_mu_min = max(t_mu - t_burnin, 0)
        y_lower_t = y_pred[t]-q[t]
        y_upper_t = y_pred[t]+q[t] #symmetric interval
        y_lowers[t] = float(y_lower_t)
        y_uppers[t] = float(y_upper_t)
        err_t = 1-float((y_lower_t <= y_test[t]) & (y_test[t] <= y_upper_t)) 
        err[t] = err_t
        smooth_err[t]= f(scores[t],q[t],np.abs(np.mean(scores[t_mu_min:t]-q[t_mu_min:t]))) if t>0 else err_t
        x = np.sum(smooth_err[:(t+1)]-alpha)
        integrator = saturation_fn_log(x, t+1, Csat, KI)
        # Update the next threshold
        if t < test_size - 1:
            if is_scorecast and t>t_burnin:
                curr_scores = np.nan_to_num(scores[:(t+1)]) 
                model = ThetaModel(curr_scores.astype(float),period=period_scorecaster).fit()
                scorecasts[t] = model.forecast(theta=2).iloc[0] #TODO modify if ahead=!1: model.forecast() forecasts the next time step
                q[t+1]=max(scorecasts[t]+integrator+lr*(smooth_err[t]-alpha),0)
            else:
                q[t+1]=max(q[t]+integrator+lr*(smooth_err[t]-alpha),0)
    print('End')
    return y_lowers, y_uppers, err, smooth_err

