import itertools

import numpy as np

from plotting import *


# self-organizing map
class SOM():

    def __init__(self, dim_in, n_rows, n_cols, attrs=None):
        self.dim_in = dim_in
        self.n_rows = n_rows
        self.n_cols = n_cols

        if attrs is None:
            self.weights = np.random.RandomState(seed=42).uniform(size=(n_rows, n_cols, dim_in))
        else:
            # init weights to match inputs
            self.weights = np.random.RandomState(seed=42).uniform(
                np.min(attrs, axis=1), np.max(attrs, axis=1), size=(n_rows, n_cols, dim_in))

    def train(self, attrs, classes, discrete=True, metric=lambda u,v:0,
              alpha_s=0.5, alpha_f=0.01, lambda_s=None, lambda_f=1,
              eps=100, in3d=False):
        (_, count) = attrs.shape

        index_2_class = dict(enumerate(np.unique(classes)))
        class_2_index = dict(map(reversed, index_2_class.items()))

        alphas = np.empty(eps)
        lambdas = np.empty(eps)
        avg_dist = np.empty(eps)
        avg_adj = np.empty(eps)
        activations = np.zeros((len(index_2_class), self.n_rows, self.n_cols))

        randomState = np.random.RandomState(seed=42)
        for ep in range(eps):

            if ep == 0:
                # prevent error
                alpha_t = alpha_s
                lambda_t = lambda_s
            else:
                exp = ep / (eps - 1)
                alpha_t  = alpha_s * ((alpha_f / alpha_s)**exp)
                lambda_t  = lambda_s * ((lambda_f / lambda_s)**exp)
            alphas[ep] = alpha_t
            lambdas[ep] = lambda_t

            print()
            print('Ep {:3d}/{:3d}:'.format(ep+1,eps))
            print('  alpha_t = {:.3f}, lambda_t = {:.3f}'.format(alpha_t, lambda_t), flush=True)

            win_distances = np.empty(count)
            adjustments = np.empty(self.n_cols)
            for i in randomState.permutation(count):
                x = attrs[:,i]

                # find winner neuron
                win_r, win_c, win_d = -1, -1, float('inf')
                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        d = np.linalg.norm(self.weights[r, c] - x)
                        if d < win_d:
                            win_r = r
                            win_c = c
                            win_d = d
                win_distances[i] = win_d
                if ep == eps - 1:
                    activations[class_2_index[classes[i]], win_r, win_c] += 1

                # learning
                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        d = metric((r,c), (win_r, win_c))
                        if discrete:
                            q = 1 if d < lambda_t else 0
                        else:
                            q = np.exp(- ((d**2) / (lambda_t**2)))
                        dC = alpha_t * (x - self.weights[r,c]) * q
                        self.weights[r,c] += dC
                        adjustments[c] = np.linalg.norm(dC)

            avg_dist[ep] = np.mean(win_distances)
            print('  quantization error = {:.3f}'.format(avg_dist[ep]))
            avg_adj[ep] = np.mean(adjustments)
            print('  average adjustment = {:.3f}'.format(avg_adj[ep]))

        # visualize training & model
        plot_training([alphas, lambdas, avg_dist, avg_adj],
                     ['alpha decay', 'lambda decay', 'quantization error', 'average adjustment'])
        plot_training([alphas, lambdas, avg_dist, avg_adj],
                     ['alpha decay', 'lambda decay', 'quantization error', 'average adjustment'],
                      cut_top=1)

        plot_attrs([self.weights[:,:,coor] for coor in range(self.dim_in)])

        dist_horizontal = np.empty((self.n_rows, self.n_cols-1))
        for r in range(self.n_rows):
            for c in range(self.n_cols-1):
                dist_horizontal[r,c] = np.linalg.norm(self.weights[r,c] - self.weights[r,c+1])
        dist_vertical = np.empty((self.n_rows-1, self.n_cols))
        for c in range(self.n_cols):
            for r in range(self.n_rows-1):
                dist_vertical[r,c] = np.linalg.norm(self.weights[r,c] - self.weights[r+1,c])
        plot_umatrix([dist_horizontal, dist_vertical], ['horizontal', 'vertical'])

        plot_activation(activations, index_2_class)

        for i_x, i_y in itertools.combinations(range(attrs.shape[0]), 2):
            plot_grid_2d(attrs, self.weights, i_x, i_y)
