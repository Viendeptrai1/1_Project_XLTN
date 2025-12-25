import numpy as np
from sklearn.decomposition import PCA
f = open('mix_01.txt', 'rt')
data = f.read()
f.close()
data = data.split()
x1 = []
for value in data:
    x1.append(int(value)/32768 + 1.0)

f = open('mix_02.txt', 'rt')
data = f.read()
f.close()
data = data.split()
x2 = []
for value in data:
    x2.append(int(value)/32768 + 1.0)
n = len(x2)
x = np.zeros((n, 2), np.float32)

for i in range(0, n):
    x[i][0] = x1[i]
    x[i][1] = x2[i]
    
m = np.mean(x, 0)
x = x - m

# Kiểm tra ma trận convariance
C = np.matmul(x.T, x)
C = C / n
print ('C', C)

pca = PCA(whiten=True)
y = pca.fit_transform(x)

# Kiểm tra ma trận convariance
C = np.matmul(y.T, y)
C = C / n
print ('C sau khi làm trắng ', C)






# C11 = 0.0
# C12 = 0.0
# C21 = 0.0
# C22 = 0.0
# for i in range(0, n):
#     C11 += x1[i] * x1[i]
#     C12 += x1[i] * x2[i]
#     C21 += x2[i] * x1[i]
#     C22 += x2[i] * x2[i]
# C11 /= n
# C12 /= n
# C21 /= n
# C22 /= n
# print('C = ')
# print(C11, C12)
# print(C21, C22)

# C = np.array([[C11, C12], [C21, C22]])
# C11 = np.sqrt(C11)
# C12 = np.sqrt(C12)
# C21 = np.sqrt(C21)
# C22 = np.sqrt(C22)

# print('C = ')
# print(C11, C12)
# print(C21, C22)


# V = np.array([[C11, C12], [C21, C22]])
# V = np.linalg.inv(V)

# V11 = V[0,0]
# V12 = V[0,1]
# V21 = V[1,0]
# V22 = V[1,1]

# print('V = ')
# print(V11, V12)
# print(V21, V22)

# y1 = []
# y2 = []
# for i in range(0, n):
#     temp = V11*x1[i] + V12*x2[i]
#     y1.append(V11*x1[i] + V12*x2[i])  
#     y2.append(V21*x1[i] + V22*x2[i]) 
    
# C11 = 0.0
# C12 = 0.0
# C21 = 0.0
# C22 = 0.0
# for i in range(0, n):
#     C11 += y1[i] * y1[i]
#     C12 += y1[i] * y2[i]
#     C21 += y2[i] * y1[i]
#     C22 += y2[i] * y2[i]
# C11 /= n
# C12 /= n
# C21 /= n
# C22 /= n
# print(C11, C12)
# print(C21, C22)                                                                                                                                                                                                                                                                       

# T = np.matmul(V,C)
# Result = np.matmul(T,V)
# print(Result)  