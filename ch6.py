##### chapter6 학습 관련 기술들

# SGD 복습
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr    # 학습률

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
