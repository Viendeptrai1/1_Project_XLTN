"""
DTW (Dynamic Time Warping) cho nhận dạng giọng nói
So khớp chuỗi có độ dài khác nhau bằng cách co giãn thời gian.
"""

import numpy as np


def dtw_distance(seq1, seq2):
    """
    Tính khoảng cách DTW giữa 2 chuỗi.
    Trả về giá trị khoảng cách (float).
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
    Tính khoảng cách DTW và đường căn chỉnh tối ưu.
    Trả về (distance, path).
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
    
    # Truy vết đường đi
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
    """Bộ phân loại dựa trên DTW cho nhận dạng giọng nói."""
    
    def __init__(self):
        self.templates = []
        self.labels = []
    
    def fit(self, X_train, y_train):
        """Lưu các mẫu huấn luyện (templates)."""
        self.templates = X_train
        self.labels = y_train
        return self
    
    def predict_single(self, x):
        """Dự đoán nhãn cho 1 chuỗi. Trả về (nhãn, khoảng_cách)."""
        min_distance = np.inf
        best_label = None
        
        for template, label in zip(self.templates, self.labels):
            distance = dtw_distance(x, template)
            if distance < min_distance:
                min_distance = distance
                best_label = label
        
        return best_label, min_distance
    
    def predict(self, X_test):
        """Dự đoán nhãn cho nhiều chuỗi. Trả về (predictions, distances)."""
        predictions = []
        distances = []
        
        for x in X_test:
            label, distance = self.predict_single(x)
            predictions.append(label)
            distances.append(distance)
        
        return predictions, distances
    
    def score(self, X_test, y_test):
        """Tính accuracy trên tập test."""
        predictions, _ = self.predict(X_test)
        correct = sum(pred == true for pred, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        return accuracy
