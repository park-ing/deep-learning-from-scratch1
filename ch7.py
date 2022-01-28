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
