# -*- coding:utf-8 -*-
import sys

if sys.version_info < (3, 0):
    reload(sys)
    sys.setdefaultencoding('utf8')
    from config import all_config
else:
    from util.config import all_config
import logging
import time
import csv


def LogInit():
    all_config.load()
    logging.basicConfig(level=logging._checkLevel(all_config['log']['filelv']),
                        format='%(asctime)s %(filename)s:%(lineno)d[%(levelname)s] %(message)s',
                        datefmt='%Y%d%m %H:%M:%S',
                        filename="./log/" + time.strftime("%Y%m%d-%H%M%S_") + all_config['log']['filepath'],
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging._checkLevel(all_config['log']['screanlv']))
    formatter = logging.Formatter('%(asctime)s %(funcName)s:%(lineno)d[%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def Draw3dPairs(point_pairs):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    x = np.array(point_pairs)[:, 0]
    y = np.array(point_pairs)[:, 1]
    z = np.array(point_pairs)[:, 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=cm.gist_earth, )

    plt.show()


def Draw2dScatterWithValue(point_pairs):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.array(point_pairs)[:, 0]
    y = np.array(point_pairs)[:, 1]
    z = np.array(point_pairs)[:, 2]

    close = z

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=close, alpha=0.3, linewidth=0.0)

    plt.show()


def DrawHist(nums, filename, label=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import mlab

    x = np.array(nums)
    total = len(x)
    n_bins = 20
    # n_bins = int(np.sqrt(len(x) * 2))
    bin_width=(x.max()-x.min())/20
    mu = x.mean()
    sigma = x.std()
    # logging.info("len(x)= %d, n_bins= %d, mu= %0.3f, sigma= %0.3f" % (len(x), n_bins, mu, sigma))
    # fig, ax = plt.subplots(figsize=(8, 4))

    # plot the histogram
    n, bins, patches = plt.hist(x, n_bins, density=False, cumulative=False, histtype='bar', rwidth=0.8, label=label,
                                alpha=0.7)

    # Add a line showing the expected distribution.
    y = mlab.normpdf(bins, mu, sigma)*bin_width*total
    plt.plot(bins, y, 'k--', linewidth=1.5)

    # tidy up the figure
    # ax.grid(True)
    # ax.legend(loc='right')
    plt.title(r'Histogram: $\mu=%0.4f$, $\sigma=%0.4f$, total=%s' % (mu, sigma, total))
    plt.xlabel('Value')
    #plt.ylabel('Number')
    plt.savefig(filename)
    plt.close()
    # plt.show()


def animation_demo():
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np
    fig, ax = plt.subplots()

    x = np.arange(0, 2 * np.pi, 0.01)
    line, = ax.plot(x, np.sin(x))

    def animate(i):
        line.set_ydata(np.sin(x + i / 10.0) * np.sin(i / 20.0) * 0.5)
        return line,

    def init():
        line.set_ydata(np.sin(x))
        return line,

    ani = animation.FuncAnimation(fig=fig,
                                  frames=1000,
                                  init_func=init,
                                  interval=20,
                                  blit=False)

    plt.show()


def Draw2DViaPCA(features, value):
    from sklearn.decomposition import PCA
    import numpy as np
    N = len(features)
    pca = PCA(n_components=2)
    X = np.array(features)
    print(X.shape)
    pca.fit(X)

    X = X + np.random.uniform(-0.2, 0.2, X.shape)
    layout2d = pca.transform(X)
    value_norm = np.array(value)
    value_norm = (value_norm - value_norm.min()) / (value_norm.max() - value_norm.min())
    print(layout2d.shape)
    print(layout2d[:10])
    pairs = [list(layout2d[i]) + list([value_norm[i]]) for i in range(N)]
    #    Draw3dPairs(pairs)
    Draw2dScatterWithValue(pairs)


def AppendToCsv(list, filename):
    try:
        with open(filename, "r") as fp:
            csv_r = csv.reader(fp)
            data = [r for r in csv_r]
    except:
        data = []
    data.append(list)
    with open(filename, "w") as fp: # TODO: 修改为wa模式
        csv_w = csv.writer(fp)
        for r in data:
            csv_w.writerow(r)


def AppendLinesToCsv(list, filename):
    try:
        with open(filename, "r") as fp:
            csv_r = csv.reader(fp)
            data = [r for r in csv_r]
    except:
        data = []
    data = data + list
    with open(filename, "w") as fp:
        csv_w = csv.writer(fp)
        for r in data:
            csv_w.writerow(r)


if (__name__ == "__main__"):
    animation_demo()
