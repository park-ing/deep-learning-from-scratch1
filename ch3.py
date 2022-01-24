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
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1) # y축 범위 지정
#plt.show()

##########################################
# sigmoid 함수 그리기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
#plt.plot(x, y)
#plt.ylim(-0.1, 1.1) # y축 범위 지정
#plt.show()

##########################################
# ReLU 함수 그리기
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
#plt.plot(x, y)
#plt.ylim(-1.0, 5.1) # y축 범위 지정
#plt.show()
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

############################################
# 3층 신경망 구현
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# 가중치를 임의로 설정
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) # w의 열의 개수만큼 은닉층 노드가 생성된다.
    network['b1'] = np.array([0.1,0.2,0.3]) # w의 열의 개수만큼 bias 설정
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2,W3) + b3
    y = a3

    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y) # [0.39442138 0.84045873]

##########################################
# 소프트맥스 함수
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a) # [1.34985881 18.17414537 54.59815003]

sum_exp_a = np.sum(exp_a)
print(sum_exp_a) # 74.1221542101633

y = exp_a / sum_exp_a
print(y) # [0.01821127 0.24519181 0.73659691] -> 원소들의 총합은 언제나 1
'''
def softmax(a): # 오버플로 보정 전 
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y 
'''

a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a))) # 소프트맥스 함수의 계산/ [nan nan nan] -> 제대로 계산되지 않는다.

c = np.max(a) # c = 1010(최대값)
print(a - c) # [0 -10 -20]

print(np.exp(a-c) / np.sum(np.exp(a-c))) # [9.99954600e-01 4.53978686e-05 2.06106005e-09] -> 제대로 계산 된다.

def softmax(a): ##### 오버플로 보정 후
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y 

t = np.array([0.3, 2.9, 4.0])
y = softmax(t)
print(y) # [0.01821127 0.24519181 0.73659691]
print(np.sum(y)) # 출력의 총합은 1

############################################
# 손글씨 숫자 인식
# MNIST
from dataset.mnist import load_mnist
from PIL import Image

# (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize = False)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)

# 이미지 하나 출력
img = x_train[0]
label = t_train[0]
print(img.shape)
img = img.reshape(28,28)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # numpy로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img.show()

#img_show(img)

###### 신경망의 추론 처리
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_train, t_train

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range (len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

########################
# 배치처리 구현
# 배치 처리 구현
x, t = get_data()
network = init_network()

batch_size = 100  # 배치 크기
accuracy_cnt = 0

for i in range (0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))