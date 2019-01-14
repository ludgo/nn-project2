import numpy as np

from som import *
from metric import *


dataset = np.loadtxt('seeds.txt').T
attrs = dataset[:-1]
classes = dataset[-1].astype(int)

(dim, count) = attrs.shape

# SOM
rows = 15
cols = 20

top_left = np.array((0, 0))
bottom_right = np.array((rows-1, cols-1))
metric = L_2
lambda_s = metric(top_left, bottom_right) * 0.5

model = SOM(dim, rows, cols, attrs)
model.train(attrs, classes, discrete=False, metric=metric, lambda_s=lambda_s, eps=100, in3d=dim>2)
