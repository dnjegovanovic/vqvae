import matplotlib.pyplot as plt
import numpy as np

def show(img, name):
    npimg = img.numpy()
    f = plt.figure(figsize=(16, 8))
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('./imgs/{}.png'.format(name))