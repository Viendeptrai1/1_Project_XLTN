"""
Dynamic Time Warping (DTW) for speech recognition
"""

import numpy as np


def dtw_distance(seq1, seq2):
    """
    Compute DTW distance between two sequences
    
    Parameters:
    -----------
    seq1 : np.ndarray
        First sequence (n_frames1, n_features)
    seq2 : np.ndarray
        Second sequence (n_frames2, n_features)
        
    Returns:
    --------
    distance : float
        DTW distance
    """
    if seq1.ndim == 1:
        seq1 = seq1.reshape(-1, 1)
    if seq2.ndim == 1:
        seq2 = seq2.reshape(-1, 1)
    
    n1, n2 = len(seq1), len(seq2)
    
    dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
    
    return dtw_matrix[n1, n2]


def dtw_path(seq1, seq2):
    """
    Compute DTW distance and optimal alignment path
    
    Parameters:
    -----------
    seq1 : np.ndarray
        First sequence
    seq2 : np.ndarray
        Second sequence
        
    Returns:
    --------
    distance : float
        DTW distance
    path : list of tuples
        Optimal alignment path
    """
    if seq1.ndim == 1:
        seq1 = seq1.reshape(-1, 1)
    if seq2.ndim == 1:
        seq2 = seq2.reshape(-1, 1)
    
    n1, n2 = len(seq1), len(seq2)
    
    dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
    
    i, j = n1, n2
    path = [(i - 1, j - 1)]
    
    while i > 0 and j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            ]
            argmin = np.argmin(candidates)
            
            if argmin == 0:
                i -= 1
            elif argmin == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        
        path.append((i - 1, j - 1))
    
    path.reverse()
    
    return dtw_matrix[n1, n2], path


class DTWClassifier:
    """
    DTW-based classifier for speech recognition
    """
    
    def __init__(self):
        self.templates = []
        self.labels = []
    
    def fit(self, X_train, y_train):
        """
        Store training templates
        
        Parameters:
        -----------
        X_train : list of np.ndarray
            Training sequences (MFCC features)
        y_train : list
            Training labels
        """
        self.templates = X_train
        self.labels = y_train
        
        return self
    
    def predict_single(self, x):
        """
        Predict label for a single sequence
        
        Parameters:
        -----------
        x : np.ndarray
            Test sequence
            
        Returns:
        --------
        label : str or int
            Predicted label
        distance : float
            Distance to nearest template
        """
        min_distance = np.inf
        best_label = None
        
        for template, label in zip(self.templates, self.labels):
            distance = dtw_distance(x, template)
            
            if distance < min_distance:
                min_distance = distance
                best_label = label
        
        return best_label, min_distance
    
    def predict(self, X_test):
        """
        Predict labels for multiple sequences
        
        Parameters:
        -----------
        X_test : list of np.ndarray
            Test sequences
            
        Returns:
        --------
        predictions : list
            Predicted labels
        distances : list
            Distances to nearest templates
        """
        predictions = []
        distances = []
        
        for x in X_test:
            label, distance = self.predict_single(x)
            predictions.append(label)
            distances.append(distance)
        
        return predictions, distances
    
    def score(self, X_test, y_test):
        """
        Compute accuracy on test set
        
        Parameters:
        -----------
        X_test : list of np.ndarray
            Test sequences
        y_test : list
            True labels
            
        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        predictions, _ = self.predict(X_test)
        
        correct = sum(pred == true for pred, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        
        return accuracy
