import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    for i, ax in enumerate(axes.flatten()):
        if i < len(axis_ranks):
            d1, d2 = axis_ranks[i]
            if d2 < n_comp:
                # initialisation de la figure
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
                ax.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
                ax.set_title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))

                # affichage des flèches
                if pcs.shape[1] < 30:
                    ax.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                       pcs[d1,:], pcs[d2,:], 
                       angles='xy', scale_units='xy', scale=1, color="grey")
                else:
                    lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                    ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

                # affichage des noms des variables  
                if labels is not None:  
                    for j,(x, y) in enumerate(pcs[[d1,d2]].T):
                        if x >= -1 and x <= 1 and y >= -1 and y <= 1:
                            ax.text(x, y, labels[j], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

                # affichage du cercle
                circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
                ax.add_artist(circle)

                # affichage des lignes horizontales et verticales
                ax.plot([-1, 1], [0, 0], color='grey', ls='--')
                ax.plot([0, 0], [-1, 1], color='grey', ls='--')

    plt.tight_layout()
    plt.show()


        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    # création de la figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    # boucle sur les plans factoriels à afficher
    for i, (d1,d2) in enumerate(axis_ranks):
        if d2 < n_comp:
            
            # sélection des axes correspondants
            ax = axes[i]
            
            # affichage des points
            if illustrative_var is None:
                ax.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    ax.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                ax.legend()

            # affichage des labels des points
            if labels is not None:
                for j, (x,y) in enumerate(X_projected[:,[d1,d2]]):
                    ax.text(x, y, labels[j],
                            fontsize='12', ha='center',va='center')
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            ax.set_xlim([-boundary,boundary])
            ax.set_ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            ax.plot([-100, 100], [0, 0], color='grey', ls='--')
            ax.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            ax.set_xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            ax.set_ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            ax.set_title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
    
    # affichage de la figure
    plt.show(block=False)


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)