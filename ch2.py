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
