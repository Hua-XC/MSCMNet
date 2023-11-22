from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np

# 加载数据集

def plot_embedding_2d(X, y, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    
   
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X)

    ckpt_dir="images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    col=[]
    color=[0,4,5,3,8,9,1,7,6,2,17,16,14,12,13,11,18,15,17,10]
    # color=['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
    #        'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys',
    #        'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    #        'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
    #        'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 
    #        'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
    #        'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn',
    #        'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 
    #        'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 
    #        'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r',
    #        'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r',
    #        'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
    #        'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 
    #        'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 
    #        'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 
    #        'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r',
    #        'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter', 
    #        'winter_r']

    for i in range(0,len(y)):
        col.append(color[y[i]%20])
        
    # XX = np.arange(-5, 5, 0.25)
    # YY = np.arange(-5, 5, 0.25)
    # XX, YY = np.meshgrid(XX, YY)
    # R = np.sqrt(XX**2 + YY**2)
    # Z = np.sin(R)  
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color=col,colormap='jet',label="t-SNE",marker='4')#hsv_r
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=col,cmap='tab20_r',label="t-SNE",marker='4',linewidth=1.1)
    plt.legend()
    # plt.subplot(122)
    # #plt.scatter(X_pca[:, 0], X_pca[:, 1], color=col,colormap='jet',label="PCA",marker='4')
    # plt.scatter(X_pca[:, 0], X_tsne[:, 1], c=col,cmap='jet',label="t-SNE",marker='4')
    # plt.legend()
    plt.savefig('images/digits_tsne-pca', dpi=1000,format='eps')
    plt.show()
    
    



