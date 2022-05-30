import numpy as np
from sklearn.decomposition import PCA

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

class PDBL_net:
    def __init__(self, isPCA, n_components, reg):
        self.reg = reg
        self.isPCA = isPCA
        self.pca = PCA(n_components=n_components,whiten=False)

    def pinv(self,A,reg):
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

    def train(self, train_in, train_out):
        if self.isPCA:
            self.pca.fit(train_in)
            train_in = self.pca.transform(train_in)
        self.A = train_in
        self.label = train_out
        self.pesuedoinverse = pinv(self.A,self.reg)
        self.W =  self.pesuedoinverse.dot(self.label)

    def predict(self, valid_in):
        if self.isPCA:
            valid_in = self.pca.transform(valid_in)
        matrix_A = valid_in
        predict = np.array(np.squeeze(matrix_A.dot(np.squeeze(self.W))))
        return predict