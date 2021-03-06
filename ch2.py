# chapter2 퍼셉트론

import numpy as np

# AND, NAND, OR 게이트
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b # w와 x 브로드캐스팅
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b # w와 x 브로드캐스팅
    if tmp <= 0:
        return 0
    else:
        return 1

# (0,0) -> 0 // (1,0) -> 1
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(w*x) + b # w와 x 브로드캐스팅
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(1,1))
print(NAND(1,1))
print(OR(1,0))

# 위의 3가지 게이트는 선형이다.
# XOR게이트 : 비선형 게이트, 단층 퍼셉트론만으론 구현할 수 없다.
# 단층 퍼셉트론을 '층을 쌓아' 다층 퍼셉트론을 구현할 수 있다. (비신형 게이트 구현 가능)
# (0,0) -> 0 / (0,1) -> 1 / (1,1) -> 0
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0,0))
print(XOR(1,0))
print(XOR(1,1))