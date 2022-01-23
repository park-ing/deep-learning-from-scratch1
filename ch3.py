# chapter3 신경망
import numpy as np
import matplotlib.pylab as plt
# 활성화 함수의 종류

################################################
# 계단함수(step function)
'''
def step_function(x): #인수 하나만 받아들인다.
    if x > 0:
        return 1
    else:
        return 0
'''

def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])
print(step_function(x))

# 그래프 그리기
x = np.arange(-5.0, 5.0, 0.1) # -5.0에서 5.0 전까지 0.1 간격의 넘파이 배열을 생성
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()

##########################################
# sigmoid 함수 그리기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()

##########################################
# ReLU 함수 그리기
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.1) # y축 범위 지정
plt.show()
##########################################

# 다차원 행렬 곱셈
A = np.array([[1,2], [3,4]])
print(A.shape)
B = np.array([[5,6], [7,8]])
print(B.shape)
print(np.dot(A, B))
print("------")
print(A*B)

# 신경망에서의 행렬 곱
print("신경망에서의 행렬 곱")
x = np.array([1,2])
print(x.shape)
w = np.array([[1,3,5],[2,4,6]])
print(w.shape)
y = np.dot(x,w)
print(y)

