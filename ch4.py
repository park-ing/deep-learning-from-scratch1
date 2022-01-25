import numpy as np
############손실 함수####################### 
# 오차 제곱 합
# '2'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 원-핫 인코딩

def sum_squares_error(y, t):      # 오차 제곱 합
    return 0.5 * np.sum((y-t)**2)

print(sum_squares_error(np.array(y), np.array(t))) # 0.09750000000000003

# '7'일 확률이 가장 높다고 추정함(0.6)
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(sum_squares_error(np.array(y), np.array(t))) # 0.5975


##### 교차 엔트로피 오차
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
# np.log를 계산할 때 아주 작은 값인 delta 추가 
# -> 0을 입력하면 마이너스 무한대 -inf가 튀어나와서 더 이상 계산 진행 불가능

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(cross_entropy_error(np.array(y), np.array(t))) # 2.302584092994546

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(cross_entropy_error(np.array(y), np.array(t))) # 0.510825457099338
# 후자의 오차가 더 작으므로 학습내용이 정답 레이블에 근접

#######################################
# 미니배치 학습
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# 이 코드에서 무작위로 10장만 빼내기

# 미니배치 학습
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 무작위 인덱스 출력
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch)
print()
print(t_batch)

print(np.random.choice(60000,10)) # [0,60000) 범위 안에서 랜덤한 인덱스 10개 출력!


#######배치용 교차 엔트로피 오차 계산하기
def cross_entropy_error(y, t):
    if y.ndim == 1:                     # y가 1차원, 즉 데이터 하나만 들어올 떄.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
     
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7)) / batch_size
# 배치의 크기로 나눠(batch_size로 나눠준다) 정규화하고 이미지 1장당 평균의 교차 엔트로피 오차를 계산한다.


''' t가 원-핫 인코딩이 아닐 때 
def cross_entropy_error(y, t):
    if y.ndim == 1:                     # y가 1차원, 즉 데이터 하나만 들어올 떄.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
     
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
# np.arange(batch_size)는 0부터 batch_size - 1까지 배열을 생성한다.
# 예를 들어 batch_size가 5이면 [0,1,2,3,4]라는 넘파이 배열을 생성한다.
# t에는 레이블이 [2,7,0,9,4]와 같이 저장되어 있으므로 y[np.arange(batch_size), t]는
# 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출한다. (ex/ y[0,2], y[1,7],y[2,0],y[3,9]..
'''
####################################
# 수치 미분
# 나쁜 구현 예
def numerical_diff(f, x):
	h = 10e-50
	return (f(x+h) - f(x)) / h

# 중앙 차분을 이용해서 식을 보정해준다.
# 이게 좀 더 정확도 높음
def numerical_diff(f, x):
	h = 1e-4 # 0.0001
	return (f(x+h) - f(x-h)) / (2*h)

def function_2(x):
	return x[0]**2 + x[1]**2

# x0에 대해 편미분
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

print(numerical_diff(function_tmp1, 3.0)) # 6.00000000000378
# 실제 해석적 편미분 값과 거의 동일함.

#################################
# 기울기
# 1개 이상의 변수를 가진 함수를 동시에 편미분한다.
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0]))) # [6. 8.]
print(numerical_gradient(function_2, np.array([0.0, 2.0]))) # [0. 4.]
print(numerical_gradient(function_2, np.array([3.0, 0.0]))) # [6. 0.]

#########경사 하강법
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad # 편미분값과 학습률을 곱해서 x값을 보정해준다.
    return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr = 0.1, step_num=100))
# [-6.11110793e-10  8.14814391e-10] -> 거의 [0, 0]과 비슷. 
# 실제 진정한 최솟값은 (0,0)이므로 경사법으로 거의 정확한 결과를 얻음.


######간단한 신경망에서 실제 가중치의 기울기 구하기
# 간단한 신경망 클래스 구현
def softmax(a): ##### 오버플로 보정 후
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y 

def numerical_gradient_mul(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
    
net = simpleNet()
print(net.W)
# [[ 0.23439824 -0.43980342  0.02929139]
#  [ 0.23084003  0.44303601 -0.68263663]]

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# [ 0.34839497  0.13485036 -0.59679813]

t = np.array([0,0,1])
print(net.loss(x, t))
# 1.7319760658246335

# 가중치의 기울기 구하기
def f(w):
    return net.loss(x, t)

dw = numerical_gradient_mul(f, net.W)
print(dw)
'''
[[ 0.23128525  0.20033183 -0.43161708]
 [ 0.34692787  0.30049775 -0.64742562]]
 '''
####################################
# 학습 알고리즘 구현하기
