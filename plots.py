import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
from matplotlib import rc
from fit_krr import Data

# Set fig size
plt.rcParams["figure.figsize"] = (16,9)
##Set plotting theme
sns.set(font_scale=2.,rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})
rc('text', usetex=False)
# Set "b", "g", "r" to default seaborn colors
sns.set_color_codes("deep")


def auto_bin(x, y, n=10):
    avg = lambda x: np.exp(np.mean(np.log(x)))
    #avg = lambda x: np.median(x)
    X, Y = [], []
    order = np.argsort(x)
    for i in range(0, x.size, n):
        idx = order[i:i+n]
        X.append(avg(x[idx]))
        Y.append(avg(y[idx]))
    return np.asarray(X), np.asarray(Y)



def plot_variance_vs_error():
    data = joblib.load("ml/data.pkl")
    predictions = joblib.load("ml/predictions.pkl")
    energies = data.energies
    errors1 = predictions[:, -1] - energies
    errors2 = predictions[:, :-1].mean(1) - energies
    var = predictions[:, :-1].std(ddof=1.5, axis=1)

    x, y = auto_bin(var, abs(errors1), n=20)
    #plt.scatter(*auto_bin(var, abs(errors2), n=50))
    plt.scatter(x, y, c="b")
    plt.show()
    quit()
    plt.xlabel("Standard deviation")
    plt.ylabel("Absolute error")
    plt.xlim(0, 37)
    plt.ylim(0, 142)
    plt.savefig("var_vs_error_binned.png", pad_inches=0.0, bbox_inches="tight", dpi=300)
    plt.clf()
    plt.scatter(var, abs(errors1), c="orange")
    plt.scatter(x, y, c="b")
    plt.xlabel("Standard deviation")
    plt.ylabel("Absolute error")
    plt.xlim(0, 37)
    plt.ylim(0, 142)
    plt.savefig("var_vs_error.png", pad_inches=0.0, bbox_inches="tight", dpi=300)

def plot_vae():
    def preprocess(d):
        y = []
        x = []
        for i, id_ in enumerate(np.unique(ids[:,0])):
            idx = np.where(ids == id_)[0]
            y.append(abs(errors[id_]))
            x.append(max(d[idx]))
        return np.asarray(x), np.asarray(y)

    data = joblib.load("ml/data.pkl")
    predictions = joblib.load("ml/predictions.pkl")
    energies = data.energies
    filenames = data.filenames
    errors = predictions[:, -1] - energies
    vae_data = joblib.load("vae_data.pkl")


    #r = (vae_data[:,0]**2 + vae_data[:,1]**2)**0.5
    #idx = [np.argmax(r)]

    #r = np.exp(vae_data[:,2]) + np.exp(vae_data[:,3])
    #r = np.exp(vae_data[:,-1])

    plt.scatter(vae_data[:,0], vae_data[:,1], alpha=0.1)
    plt.gca().set_aspect('equal')
    plt.savefig("vae.png", pad_inches=0.0, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    #plot_variance_vs_error()
    plot_vae()
