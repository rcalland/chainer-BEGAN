from chainer import Variable
import numpy as np

def func(x):
    x = x + 1.0

x = Variable(np.array(0.0))

func(x)
print(x.data)
