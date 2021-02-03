import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
from matplotlib import rc

# Set fig size
plt.rcParams["figure.figsize"] = (16,9)
##Set plotting theme
sns.set(font_scale=2.,rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})
rc('text', usetex=False)
# Set "b", "g", "r" to default seaborn colors
sns.set_color_codes("deep")


def plot_vae():
    vae_data = joblib.load("vae_data.pkl")

    #plt.scatter(vae_data[:103,0], vae_data[:103,1], alpha=0.2)
    #plt.scatter(vae_data[103:184,0], vae_data[103:184,1], alpha=0.2)
    #plt.scatter(vae_data[184:,0], vae_data[184:,1], alpha=0.2)
    plt.scatter(vae_data[:,0], vae_data[:,1], alpha=0.2)
    plt.gca().set_aspect('equal')
    plt.savefig("vae.png", pad_inches=0.0, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    plot_vae()
