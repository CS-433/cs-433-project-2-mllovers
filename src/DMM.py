import torch
from torch.nn import Module, Parameter
from torch.nn.functional import softmax, log_softmax
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from visualization_helpers import plot_by_groups

class DirichletMixture(Module):
    def __init__(self, M, sparsity=0, D=12, transposition=True):
        super(DirichletMixture, self).__init__()
        """
        Parameters:
            - M: int scalar 
            Number of clusters
            - sparsity: float scalar 
            Regularization constant
            - D: int scalar 
            Dimension of space
            - transposition: bool 
            if True then model with transposition invariance is used
        """
        # type of float tensors
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # parameters of Dirichlet model
        self.alphas = Parameter(torch.rand(M, D).type(dtype))
        # parameters of mixture(mixture weights) 
        self.w = Parameter(torch.ones(M, 1).type(dtype))
        # sparsity regularization constant
        self.sparsity = sparsity #[float]
        # if true, then model with transposition invariance is used
        self.transpostition = transposition # [bool]

    def param(self, roll=0):
        """
        Parameters:
            - roll: int. Default: 0
            Number of transposition
        Returns: tensor (number of clusters, number of features)
        Parameters of Dirichlet model (have to be > 0)  
        """
        return torch.roll(self.alphas**2, roll, dims=1)
    
    def norm_log(self, roll=0):
        """
        Logarithm of the normalization constant of the Dirichlet distribution for given parameters
        Parameters:
            - roll: int. Default: 0
            Number of transposition 
        Returns: tensor (number of clusters, 1)
        """
        norm = - torch.lgamma(self.param(roll)).sum(1) + torch.lgamma(self.param(roll).sum(1))
        return norm.reshape(-1,1)
    
    def weights_log(self):
        """
        Logarithm of the weight in mixture model for log-likelihood
        Returns: tensor (number of clusters, 1)
        """
        return log_softmax(self.w, 0)

    def sum_over_transpositions(self, sample):
        """
        Sum over transpositions for obtaining log-likelihood
        Parameters:
            - sample: tensor (number of samples, number of features)
            Input data(pitch scapes)
        Returns: tensor (number of clusters, number of samples)
        """
        sum_ = 0 # intermediate variable
        Num_transpos = 12 # Number of possible transpositions
        for roll in range(Num_transpos):
            alphas = self.param(roll)
            sum_ += torch.exp(torch.matmul(alphas-torch.ones_like(alphas), torch.log(sample).T) + self.norm_log(roll))
        return sum_/Num_transpos
    
    
    def likelihoods(self, sample):
        """
        Likelihood for given sample
        Parameters:
            - sample: tensor (number of samples, number of features)
            Input data(pitch scapes)
        Returns: tensor (number of clusters, number of samples)
        likelihood(sample)
        """
        alphas = self.param()
        if self.transpostition:
            return torch.exp(torch.log(self.sum_over_transpositions(sample)) + self.weights_log())
        else:
            return torch.exp(torch.matmul(alphas-torch.ones_like(alphas), torch.log(sample).T) + self.weights_log() + self.norm_log())
        
    def log_likelihoods(self, sample):
        """
        Log-likelihood for given sample
        Parameters:
            - sample: tensor (number of samples, number of features)
            Input data(pitch scapes)
        Returns: tensor (number of samples)
        log(likelihood(sample))
        """
        alphas = self.param()
        if self.transpostition:
            return torch.log(torch.exp(torch.log(self.sum_over_transpositions(sample)) + self.weights_log()).sum(0))
        else:
            return torch.log(torch.exp(torch.matmul(alphas-torch.ones_like(alphas), torch.log(sample).T) + self.weights_log() + self.norm_log()).sum(0))
        
    def neglog_likelihood(self, sample):
        """
        Loss function to minimize
        Parameters:
            - sample: tensor (number of samples, number of features)
            Input data(pitch scapes)
        Returns: scalar
        -log(likelihood(sample)) with regularization term
        """
        ll = self.log_likelihoods(sample)
        log_likelihood = torch.mean(ll)
        return -log_likelihood + self.sparsity * softmax(self.w, 0).sqrt().mean()
    

    def predict(self, sample):
        """
        Infer labels for samples
        Parameters:
            - sample: tensor (number of samples, number of features)
            Input data(pitch scapes)
        Returns: tensor (number of samples)
        Predicted labels by model for sample
        """
        label = torch.argmax(self.likelihoods(sample), dim=0)
        return label

    def plot_clusters(self, sample, labels, dim=2, size=4, name='clusters_visualization', opacity=0.9):
        """
        Plot clusters of samples according to predicted labels
        Parameters:
            - sample: tensor (number of samples, number of features)
              Input data (pitch scapes)
            - labels: tensor (number of samples)
              Oredicted labels for input data
            - dim: int, 2 or 3
              A parameter that defines the dimension of the projected data. Default: 2 
            - size: int
              A parameter that defines the size of points. Default: 4 
            - name: str
              A name that is used for saving the plot. If None is given, it doesn't save the plot.
              Default: None
            - opacity: float or array-like (number of samples, )
              A parameter is in the interval [0, 1] that determines the visibility of points. 
              Default: 1.0
        """
        plot_by_groups(sample.detach().cpu().numpy(), labels.detach().cpu().numpy(), \
                       type_name='cluster', opacity=opacity, name=name, size=size, dim=dim)
