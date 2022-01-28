# chapter7 합성곱 신경망(CNN)
import numpy as np
################
# 4차원 배열
x = np.random.rand(10,1,28,28) # 무작위로 데이터 생성
print(x.shape)
#print(x)
# 첫 번째 데이터에 접근
#print(x[0])
# 첫 번째 데이터의 첫 채널의 공간 데이터에 접근
#print(x[0,0]) # 또는 x[0][0]

############################
# 합성곱 계층 구현하기
from common.util import im2col
## im2col(input_data, filter_h, filter_w, stride=1,pad=0)
#   input_data : (데이터 수, 채널 수, 높이 너비)의 4차원 배열로 이뤄진 입력 데이터
#   filter_h : 필터의 높이
#   filter_w ; 필터의 너비
#   stride : 스트라이드
#   pad : 패딩

x1 = np.random.rand(1,3,7,7) # (데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1,pad=0)
print(col1.shape)   # (9, 75) 

x2 = np.random.rand(10,3,7,7) # (데이터 수, 채널 수, 높이, 너비)
col2 = im2col(x2, 5, 5, stride=1,pad=0)
print(col2.shape)   # (90,75)
# col 두번째 차원의 원소 수 -> 채널3개 x 5x5데이터 = 75

# 합성곱 계층 Convolution 구현
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T    # 필터 전개
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
# 풀링 계층 구현
# 1. 입력 데이터를 전개한다.
# 2. 행별로 최대값을 구한다.
# 3. 적절한 모양으로 성형한다.
class Pooling:
    def __init__ (self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 최대값(2)
        out = np.max(col, axis=1)

        # 성형(3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out


