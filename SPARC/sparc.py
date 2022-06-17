import graphtools
from scipy.spatial.distance import pdist, squareform, jensenshannon
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import tasklogger
import warnings
warnings.simplefilter('ignore')

# Creating class

class spARC(object):
    def __init__(
        self,
        expression_graph = None,
        spatial_graph = None,
        expression_knn=15,
        spatial_knn=15,
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
        
        self.soluable_spatial_graph = None
        self.soluable_diff_op = None
        self.soluable_diff_op_powered = None
        self.spatial_t_soluable = 10
        self.X_sparc = None
        self.X_sparc_soluable = None
        
        
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
        
        adjacency = np.zeros(self.X_pca.shape[0]**2).reshape(self.X_pca.shape[0],self.X_pca.shape[0])
        with tasklogger.log_task("spatial graph"):
            if self.spatial_graph == None:
                
                spatial_neighbors = NearestNeighbors(n_neighbors=self.spatial_knn+1,
                                                     algorithm="ball_tree").fit(self.spatial_X).kneighbors(return_distance=False)

                rna_neighbors = NearestNeighbors(n_neighbors=3*self.spatial_knn+1,
                                                 algorithm="ball_tree").fit(self.X_pca).kneighbors(return_distance=False)
                
                #adjacency = np.zeros(self.X_pca.shape[0]**2).reshape(self.X_pca.shape[0],self.X_pca.shape[0])
                for r in range(spatial_neighbors.shape[0]):
                    intersection = np.intersect1d(spatial_neighbors[r,:], rna_neighbors[r,:])
                    adjacency[r,intersection] = adjacency[r,intersection] + 1   
                adjacency = adjacency + adjacency.T
                
                self.spatial_graph = graphtools.Graph(adjacency, n_pca = None, precomputed='adjacency',
                                                      decay=self.spatial_decay, verbose = False,
                                                      n_jobs=self.n_jobs,random_state = self.random_state)
            self.spatial_diff_op = self.spatial_graph.diff_op
        return
            
            

    def transform(self):
        if self.expression_diff_op_powered == None:
            with tasklogger.log_task("random walks on expression graph"):
                self.expression_diff_op_powered = np.linalg.matrix_power(self.expression_diff_op.toarray(),
                                                                        self.expression_t)
        
        if self.spatial_diff_op_powered == None:
            with tasklogger.log_task("random walks on spatial graph"):
                self.spatial_diff_op_powered = np.linalg.matrix_power(self.spatial_diff_op,
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
            self.soluable_spatial_graph = graphtools.Graph(self.spatial_X, n_pca = None,
                                                           distance=self.knn_dist, knn = 2*self.spatial_knn+1,
                                                           decay=self.expression_decay,n_jobs=self.n_jobs,
                                                           random_state = self.random_state, verbose = False)
            self.soluable_diff_op = self.soluable_spatial_graph.diff_op
            
        with tasklogger.log_task("diffusion on soluable factors"):
            data_sparc_ligands = self.X_sparc.copy()
            self.X_sparc_soluable = self.X_sparc.copy()
            for t in range(soluable_t):
                data_sparc_ligands = self.soluable_diff_op @ data_sparc_ligands
                self.X_sparc_soluable = data_sparc_ligands + self.X_sparc_soluable
        return self.X_sparc_soluable

def compress_data(X, n_pca=50, random_state=None):
    pca_op = sklearn.decomposition.PCA(n_components=n_pca, random_state=random_state)
    return pca_op.fit_transform(X), pca_op