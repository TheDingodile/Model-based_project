import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
from model import Net, FFNet
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
from pyro.infer import Predictive

def standardize_and_split(X,y):

    # standardize input features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    # standardize target
    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std

    # Split data into traning, test and validation
    training_idx = (data['Year'] >= 2000) & (data['Year'] <= 2011)
    test_idx = (data['Year'] >= 2012) & (data['Year'] <= 2013)
    val_idx = (data['Year'] >= 2014) & (data['Year'] <= 2015)

    X_train = X[training_idx,:]
    X_test = X[test_idx,:]
    X_val = X[val_idx, :]

    y_train = y[training_idx]
    y_test = y[test_idx]
    y_val = y[val_idx]

    return X_train, X_test, X_val, y_train, y_test, y_val

def compute_rmse(preds_train,preds_test,preds_val, y_train, y_test, y_val, y_std, y_mean):
    # Convert back to the original scale
    y_true_train = y_train * y_std + y_mean
    y_true_test = y_test * y_std + y_mean
    y_true_val = y_val * y_std + y_mean
    def rmse(predicted,trues):
        rmse = np.sqrt(np.mean((predicted - trues)**2))
        return rmse
    rmse_train = rmse(y_true_train, preds_train)
    rmse_test = rmse(y_true_test, preds_test)
    rmse_val = rmse(y_true_val, preds_val)
    print("RMSE train: %.3f\nRMSE test: %.3f\nRMSE val: %.3f\n" % (rmse_train,rmse_test,rmse_val))
    return rmse_train, rmse_test, rmse_val

def eval_regression(regr,y_train, y_test, y_val,y_std,y_mean):
    y_hat_train = regr.predict(X_train)*y_std+y_mean
    y_hat_test = regr.predict(X_test)*y_std+y_mean
    y_hat_val = regr.predict(X_val)*y_std+y_mean
    return compute_rmse(y_hat_train,y_hat_test,y_hat_val,y_train,y_test,y_val,y_std,y_mean)

def eval_bayesian(beta_samples,X_train,X_test,X_val,y_train, y_test, y_val,y_std,y_mean):
    y_hat_train =  np.mean(np.dot(X_train, beta_samples[:,0].T), axis=1)*y_std+y_mean
    y_hat_test = np.mean(np.dot(X_test, beta_samples[:,0].T), axis=1)*y_std+y_mean
    y_hat_val = np.mean(np.dot(X_val, beta_samples[:,0].T), axis=1)*y_std+y_mean
    return compute_rmse(y_hat_train,y_hat_test,y_hat_val,y_train, y_test, y_val,y_std,y_mean)

def train_linear_regression(X_train,y_train,features):
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(X_train, y_train)
    for c, f in zip(regr.coef_,features):
        print("Feature: {} = {}".format(f,c))
    return regr

def bayesian_regression(X_train,y_train,features):

    def model(X, obs=None):
        beta  = pyro.sample("beta", dist.Normal(torch.zeros(X.shape[1]), 
                                                torch.ones(X.shape[1])).to_event())    # Priors for the regression coeffcients
        sigma = pyro.sample("sigma", dist.HalfCauchy(1.0))                   # Prior for the variance
        
        with pyro.plate("data"):
            y = pyro.sample("y", dist.Normal(X.matmul(beta), sigma), obs=obs)
            
        return y

    # Define guide function
    guide = AutoMultivariateNormal(model)

    # Reset parameter values
    pyro.clear_param_store()

    # Define the number of optimization steps
    n_steps = 5000

    # Setup the optimizer
    adam_params = {"lr": 0.001} # learning rate (lr) of optimizer
    optimizer = ClippedAdam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=1)
    svi = SVI(model, guide, optimizer, loss=elbo)

    X_train_torch = torch.tensor(X_train).float()
    y_train_torch = torch.tensor(y_train).float()

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step(X_train_torch, y_train_torch)
        if step % 100 == 0:
            print("[%d] ELBO: %.1f" % (step, elbo))

    predictive = Predictive(model, guide=guide, num_samples=1000,
                        return_sites=("beta", "sigma"))
    samples = predictive(X_train_torch, y_train_torch)
    beta_samples = samples["beta"].detach().numpy()
    betas = beta_samples.mean(axis=0)

    for c, f in zip(betas[0,:],features):
        print("Feature: {} = {}".format(f,c))

    return beta_samples

def train_neural_network(X_train,y_train, model,linear=False):
    X_train_torch = torch.tensor(X_train).float()
    y_train_torch = torch.tensor(y_train).float()

    # Define guide function
    guide = AutoDiagonalNormal(model)
    pyro.clear_param_store()

    # Define the number of optimization steps
    n_steps = 20000

    # Setup the optimizer
    adam_params = {"lr": 0.001}
    optimizer = Adam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=1)
    svi = SVI(model, guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step(X_train_torch, y_train_torch)
        if step % 500 == 0:
            print("[%d] ELBO: %.1f" % (step, elbo))

    X_test_torch = torch.tensor(X_test).float()
    X_val_torch = torch.tensor(X_val).float()

    # Make predictions for test set
    predictive = Predictive(model, guide=guide, num_samples=1000,
                            return_sites=("obs", "_RETURN"))
    samples_test = predictive(X_test_torch)
    samples_train = predictive(X_train_torch)
    samples_val = predictive(X_val_torch)

    y_pred_train = samples_train["obs"].mean(axis=0).detach().numpy() * y_std + y_mean
    y_pred_test = samples_test["obs"].mean(axis=0).detach().numpy() * y_std + y_mean
    y_pred_val = samples_val["obs"].mean(axis=0).detach().numpy() * y_std + y_mean

    if linear:
        predictive = Predictive(model, guide=guide, num_samples=1000,
                        return_sites=("beta",))
        samples = predictive(X_train_torch, y_train_torch)
        betas = samples["beta"].mean(axis=0).detach().numpy()[0]
        return y_pred_train, y_pred_test, y_pred_val, betas

    return y_pred_train, y_pred_test, y_pred_val


if __name__=="__main__":
    # fix random generator seed (for reproducibility of results)
    np.random.seed(42)
    data = pd.read_csv("/Users/johannesreiche/Library/Mobile Documents/com~apple~CloudDocs/DTU/MMC/Semester 4/Model-based ML/Project/Model-based_project/Data/Data_processed.csv",delimiter=";")

    mat = data.values

    features = data.columns.to_list()[5:]
    X_numpy = np.array(mat[:,5:],float)
    y_numpy = np.array(mat[:,2],float) 
    y_mean = y_numpy.mean()
    y_std = y_numpy.std()
    X_train, X_test, X_val, y_train, y_test, y_val = standardize_and_split(X_numpy,y_numpy)

    # Train and evaluate linear regression model
    regr = train_linear_regression(X_train,y_train,features)
    rmse_train, rmse_test, rmse_val = eval_regression(regr,y_train, y_test, y_val,y_std,y_mean)

    # Train and evaluate Baysian linear regression model
    beta_samples = bayesian_regression(X_train,y_train,features)
    rmse_train_b, rmse_test_b, rmse_val_b = eval_bayesian(beta_samples,X_train,X_test,X_val,y_train, y_test, y_val,y_std,y_mean)

    # Train DNN
    net = Net(n_in=X_train.shape[1], n_hidden=32, n_out=1)
    y_pred_train, y_pred_test, y_pred_val = train_neural_network(X_train,y_train,net)
    rmse_train_n, rmse_test_n, rmse_val_n = compute_rmse(y_pred_train,y_pred_test,y_pred_val, y_train, y_test, y_val, y_std, y_mean)

    # Train our combined model
    linear_features_H = [2,3,5,7,8]
    net_lr = FFNet(n_in=X_train.shape[1]-len(linear_features_H), 
                   n_hidden=32, n_out=1,linear_features=linear_features_H)
    y_pred_train2, y_pred_test2, y_pred_val2, betas = train_neural_network(X_train,y_train,net_lr,True)
    for c, f in zip(betas,np.array(features)[linear_features_H]):
        print("Feature: {} = {}".format(f,c))
    rmse_train_nl, rmse_test_nl, rmse_val_nl = compute_rmse(y_pred_train2,y_pred_test2,y_pred_val2, y_train, y_test, y_val, y_std, y_mean)