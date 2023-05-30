import torch
import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam

from pyro.nn import PyroModule, PyroSample
import torch.nn as nn

class Net(PyroModule):
    def __init__(self, n_in, n_hidden, n_out):
        super(Net, self).__init__()
        
        # Architecture
        self.in_layer = PyroModule[nn.Linear](n_in, n_hidden)
        self.in_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_in]).to_event(2))

        self.h_layer = PyroModule[nn.Linear](n_hidden, n_hidden)
        self.h_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_hidden]).to_event(2))

        self.h_layer2 = PyroModule[nn.Linear](n_hidden, n_hidden)
        self.h_layer2.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_hidden]).to_event(2))

        self.out_layer = PyroModule[nn.Linear](n_hidden, n_out)
        self.out_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_out, n_hidden]).to_event(2))

        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, X_nn, y=None):
        X_nn = self.relu(self.in_layer(X_nn))
        X_nn = self.relu(self.h_layer(X_nn))
        X_nn = self.relu(self.h_layer2(X_nn))
        X_nn = self.out_layer(X_nn)

        prediction_mean = X_nn.squeeze(-1)
        with pyro.plate("observations"):
            y = pyro.sample("obs", dist.Normal(prediction_mean, 0.1), obs=y)
            
        return y
    
class FFNet(PyroModule):
    def __init__(self, n_in, n_hidden, n_out, linear_features):
        super(FFNet, self).__init__()
        
        self.linear_features = linear_features
        self.non_linear_features = [i for i in range(15) if not i in(self.linear_features)]

        # Architecture
        self.in_layer = PyroModule[nn.Linear](n_in, n_hidden)
        self.in_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_in]).to_event(2))

        self.h_layer = PyroModule[nn.Linear](n_hidden, n_hidden)
        self.h_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_hidden]).to_event(2))

        self.h_layer2 = PyroModule[nn.Linear](n_hidden, n_hidden)
        self.h_layer2.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden, n_hidden]).to_event(2))

        self.out_layer = PyroModule[nn.Linear](n_hidden, n_out)
        self.out_layer.weight = PyroSample(dist.Normal(0., 1.).expand([n_out, n_hidden]).to_event(2))

        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, X, y=None):
        X_nn = X[:,self.non_linear_features]
        X_nn = self.relu(self.in_layer(X_nn))
        X_nn = self.relu(self.h_layer(X_nn))
        X_nn = self.relu(self.h_layer2(X_nn))
        X_nn = self.out_layer(X_nn)
        nn_out = X_nn.squeeze(-1)
        
        beta_lin = pyro.sample("beta", dist.Normal(torch.zeros(len(self.linear_features)), 
                                            torch.ones(len(self.linear_features))).to_event(1))
        X_linear = X[:,self.linear_features]
        with pyro.plate("observations"):
            linear_out = X_linear.matmul(beta_lin)
            out = nn_out+linear_out
            y = pyro.sample("obs", dist.Normal(out, 0.1), obs=y)
            
        return y