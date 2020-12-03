import torch
from torch.nn import Module, Parameter
from torch.nn.functional import softmax, log_softmax
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

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

    def plot_clusters(self, sample, labels, dims=2, metric='minkowski', text_data=None):
        """
        Plot clusters of samples according to predicted labels
        Parameters:
            - sample: tensor (number of samples, number of features)
            Input data(pitch scapes)
            - labels: tensor (number of samples)
            Oredicted labels for input data
            - dims: int scalar = {2,3}. Default: 2
            Value of dimension to map for Isomap
            - metric: str. Default: 'minkowski'
            metric of closeness for Isomap to use
            - text_data: str (number of samples). Default: None
            Additional data to print on a graph for each point with dimension number of points
        """
        fig = plt.figure(figsize=(15, 8))
        embedding = Isomap(n_components=dims, metric=metric) # initialize Isomap algorithm
        X = sample.detach().cpu().numpy() # samples have to be numpy array-like, not tensor 
        X = embedding.fit_transform(X) # Fit Isomap
        labels = labels.detach().cpu().numpy() # labels have to be numpy array-like, not tensor
        if dims == 2:
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral) # Map to 2d plane
            plt.show()
        else:
            df = pd.DataFrame({'cluster':labels, 'x':X[:, 0], 'y':X[:, 1], 'z':X[:, 2]})
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', text=text_data) # Map to 3d space
            fig.show()