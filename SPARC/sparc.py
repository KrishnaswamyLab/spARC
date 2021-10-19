import numpy as np
import graphtools
import tasklogger
import collections
from collections import defaultdict
from scipy.spatial.distance import pdist, cdist, squareform
import sklearn
import warnings

from . import vne



class spARC(object):
    """
    @dev spARC 
    """
    def __init__(
        self,
        expression_graph = None,
        spatial_graph = None,
        expression_knn=5,
        spatial_knn=2,
        expression_n_pca=50,
        expression_decay=40,
        spatial_decay=40,
        expression_t = 3,
        spatial_t = 1,
        knn_dist='euclidean',
        n_jobs=1,
        random_state=None,
        
    ):
        
        self.expression_graph = expression_graph
        self.spatial_graph = spatial_graph
        self.expression_knn = expression_knn
        self.spatial_knn = spatial_knn
        self.expression_n_pca = expression_n_pca
        self.expression_decay=expression_decay
        self.spatial_decay=spatial_decay
        self.expression_t = expression_t
        self.spatial_t = spatial_t
        self.knn_dist=knn_dist
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.expression_X = None
        self.X_pca = None
        self.pca_op = None
        self.spatial_X = None
        self.expression_diff_op = None
        self.spatial_diff_op = None
        self.expression_diff_op_powered = None
        self.spatial_diff_op_powered = None
        self.spatial_diff_op_powered_soluable = None
        self.spatial_t_soluable = 10
        self.X_sparc = None
        self.X_sparc_soluable = None
        self.soluable_spatial_graph = None
        self.soluable_diff_op = None
        
        super().__init__()

    def fit(self, expression_X, spatial_X=None):
        self.expression_X = expression_X
        self.spatial_X = spatial_X
        if self.expression_n_pca != None:
            with tasklogger.log_task("PCA"):
                 self.X_pca, self.pca_op = compress_data(self.expression_X,
                                                         n_pca=self.expression_n_pca,
                                                         random_state = self.random_state)
        with tasklogger.log_task("expression graph"):
            if self.expression_graph == None:
                self.expression_graph = graphtools.Graph(self.X_pca, n_pca = None,
                                                         distance=self.knn_dist, knn = self.expression_knn,
                                                         decay=self.expression_decay,n_jobs=self.n_jobs,
                                                         random_state = self.random_state, verbose = False)
            self.expression_diff_op = self.expression_graph.diff_op
        with tasklogger.log_task("spatial graph"):
            if self.spatial_graph == None:
                self.spatial_graph = graphtools.Graph(self.spatial_X, n_pca = None, distance='euclidean',
                                                      knn = self.spatial_knn, decay=self.spatial_decay,
                                                      n_jobs=self.n_jobs,random_state = self.random_state,
                                                      verbose = False)
            self.spatial_diff_op = self.spatial_graph.diff_op
        return
            
            

    def transform(self):
        if self.expression_diff_op_powered == None:
            with tasklogger.log_task("random walks on expression graph"):
                self.expression_diff_op_powered = np.linalg.matrix_power(self.expression_diff_op.toarray(),
                                                                        self.expression_t)
        
        if self.spatial_diff_op_powered == None:
            with tasklogger.log_task("random walks on spatial graph"):
                self.spatial_diff_op_powered = np.linalg.matrix_power(self.spatial_diff_op.toarray(),
                                                                      self.spatial_t)
        with tasklogger.log_task("spARCed expression data"):
            self.X_sparc = self.expression_diff_op_powered @ self.spatial_diff_op_powered @ self.expression_X

        return self.X_sparc
    
    def fit_transform(self, expression_X, spatial_X=None):
        with tasklogger.log_task("spARC"):
            self.fit(expression_X = expression_X, spatial_X = spatial_X)
            _ = self.transform()
        return self.X_sparc
    

    def diffuse_soluable_factors(self, soluable_t = 10, soluable_spatial_graph=None):
        self.spatial_t_soluable = soluable_t
        if soluable_spatial_graph != None:
            self.soluable_spatial_graph = soluable_spatial_graph
            self.soluable_diff_op = self.soluable_spatial_graph.diff_op
            
        else:
            self.soluable_spatial_graph = self.spatial_graph
            self.soluable_diff_op = self.soluable_spatial_graph.diff_op
            
        with tasklogger.log_task("random walks on spatial graph"):
            self.spatial_diff_op_powered_soluable = np.linalg.matrix_power(self.soluable_diff_op.toarray(),
                                                                           self.spatial_t_soluable)
        with tasklogger.log_task("diffusion on soluable factors"):
            self.X_sparc_soluable = self.spatial_diff_op_powered_soluable @ self.X_sparc
        return self.X_sparc_soluable

def compress_data(X, n_pca=50, random_state=None):
    pca_op = sklearn.decomposition.PCA(n_components=n_pca, random_state=random_state)
    return pca_op.fit_transform(X), pca_op