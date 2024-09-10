import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def cal_vina_terms(D, vdw_pair, hydro_pair, hbond_pair):
    d = D
    zero_tensor = torch.tensor(0.).to(d)
    one_tensor = torch.tensor(1.).to(d)
    #
    vdw0 = torch.exp(-4. * d * d)
    vdw1 = torch.exp(-0.25 * (d - 3.) * (d - 3.))
    vdw2 = torch.where(d < 0., d * d, zero_tensor)
    #
    hydro_mask = torch.where(d >= 1.5, zero_tensor, one_tensor)
    hydro = torch.where(d <= 0.5, one_tensor, 1.5 - d) * hydro_mask * hydro_pair
    #
    hbond_mask = torch.where(d >= 0., zero_tensor, one_tensor)
    hbond = torch.where(d <= -0.7, one_tensor, d * -1.4285) * hbond_mask * hbond_pair
    # output
    terms = [vdw0, vdw1, vdw2, hydro, hbond]
    terms = torch.cat(terms, dim=-1)
    return terms


def draw(x: np.array, y: np.array):
    import seaborn as sns
    x = x[:, 0]
    df = pd.DataFrame({'x': x, 'vdw0': y[:, 0], 'vdw1': y[:, 1], 'vdw2': y[:, 2], 'hydro': y[:, 3], 'hbond': y[:, 4]})
    sns.lineplot(data=df, x='x', y='vdw0', label='vdw0')
    sns.lineplot(data=df, x='x', y='vdw1', label='vdw1')
    sns.lineplot(data=df, x='x', y='vdw2', label='repulsive')
    sns.lineplot(data=df, x='x', y='hydro', label='hydro')
    sns.lineplot(data=df, x='x', y='hbond', label='hbond')
    plt.xlabel('d')
    plt.ylabel('score')
    plt.legend(title='terms')
    plt.ylim(y1, y2)
    plt.show()


if __name__ == '__main__':
    y1, y2 = -0.1, 0.1
    d = torch.linspace(-1, 3.5, 100).view(-1, 1)
    h1 = torch.ones_like(d)
    zero = torch.zeros_like(d)
    vina_score = cal_vina_terms(d, None, h1, h1)
    # weight
    vina_terms_weights = torch.tensor([-.0356, -.00516, 0.84, -.0351, -.587]).view(1, 5)
    score = (vina_score * vina_terms_weights).numpy()
    draw(d.numpy(), score)
    # final = (vina_score * vina_terms_weights).sum(dim=-1, keepdims=True)
    print('done')
