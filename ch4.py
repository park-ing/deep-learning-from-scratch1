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