import pandas as pd
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')

  

def calcul_metrics_kmeans(data, dataframe_metrique, type_donnees,
                          random_seed, n_clusters, n_init, init):
 
    silhouette = []
    #dispersion = []
    calin_harab = []
    davies_bouldin = []
    donnees = []
    temps = []

    result_clusters = []
    result_ninit = []
    result_type_init = []

    # Hyperparametre tuning
    n_clusters = n_clusters
    nbr_init = n_init
    type_init = init

    # Test du modèle
    for num_clusters in n_clusters:

        for init in nbr_init:

            for i in type_init:
                # Début d'exécution
                time_start = time.time()

                # Initialisation de l'algorithme
                cls = KMeans(n_clusters=num_clusters,
                             n_init=init,
                             init=i,
                             random_state=random_seed)

                # Entraînement de l'algorithme
                cls.fit(data)

                # Prédictions
                preds = cls.predict(data)

                # Fin d'exécution
                time_end = time.time()

                # Calcul du score de coefficient de silhouette
                silh = silhouette_score(data, preds)
                # Calcul de la dispersion
                #disp = cls.inertia_
                # Calcul de l'indice calinski-harabasz
                cal_har = calinski_harabasz_score(data, preds)
                # Calcul de l'indice davies-bouldin
                db = davies_bouldin_score(data, preds)
                # Durée d'exécution
                time_execution = time_end - time_start

                silhouette.append(silh)
                #dispersion.append(disp)
                calin_harab.append(cal_har)
                davies_bouldin.append(db)
                donnees.append(type_donnees)
                temps.append(time_execution)

                result_clusters.append(num_clusters)
                result_ninit.append(init)
                result_type_init.append(i)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Data type': donnees,
        'n_clusters': result_clusters,
        'n_init': result_ninit,
        'type_init': result_type_init,
        'silhouette': silhouette,
        #'dispersion': dispersion,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        #'Time (s)': temps
    }), ignore_index=True)

    return dataframe_metrique


def calcul_metrics_cah(data, dataframe_metrique, type_donnees,
                       random_seed, param_grid):
    
    silhouette = []
    davies_bouldin = []
    calin_harab = []
    donnees = []
    temps = []

    result_nclusters = []
    result_linkage = []
    result_affinity = []

    # Hyperparametre tuning
    n_clusters = param_grid[0]
    linkage = param_grid[1]
    affinity = param_grid[2]

    # Test du modèle
    for nb_clusters in n_clusters:

        for link in linkage:

            for affinite in affinity:

                # Début d'exécution
                time_start = time.time()

                # Initialisation de l'algorithme
                cah = AgglomerativeClustering(n_clusters=nb_clusters,
                                              linkage=link,
                                              affinity=affinite)

                # Entraînement de l'algorithme / Prédictions
                preds = cah.fit_predict(data)

                # Fin d'exécution
                time_end = time.time()

                # Calcul du score de coefficient de silhouette
                silh = silhouette_score(data, preds)
                # Calcul de l'indice davies-bouldin
                db = davies_bouldin_score(data, preds)
                # Calcul de l'indice calinski-harabasz
                cal_har = calinski_harabasz_score(data, preds)
                # Durée d'exécution
                time_execution = time_end - time_start

                silhouette.append(silh)
                davies_bouldin.append(db)
                calin_harab.append(cal_har)
                donnees.append(type_donnees)
                temps.append(time_execution)

                result_nclusters.append(nb_clusters)
                result_linkage.append(link)
                result_affinity.append(affinite)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'data type': donnees,
        'n_clusters': result_nclusters,
        'linkage': result_linkage,
        'affinity': result_affinity,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calin_harab,
        #'time (s)': temps
    }), ignore_index=True)

    return dataframe_metrique



  


def viz_clusters_score(dataframe, nclust_deb, nclust_fin, silh=True, dav_bou=True, cal_har=True):
    
    if silh:
        # Trace le graphique des coefficients de silhouette
        plt.figure(figsize=(15, 5))

        s_silhouette = dataframe.groupby('n_clusters')['silhouette'].mean()

        plt.title('Silhouette score vs. K', fontsize=12)
        plt.plot([i for i in range(nclust_deb, nclust_fin)], s_silhouette,
                 linestyle="--", marker='o')
        plt.grid(False)
        plt.xlabel('K', fontsize=12)
        plt.ylabel('Silhouette score', fontsize=12)
        plt.xticks([i for i in range(nclust_deb, nclust_fin)], fontsize=11)
        plt.yticks(fontsize=11)

        plt.show()

    if dav_bou:
        # Trace le graphique des indices de Davies-Bouldin
        plt.figure(figsize=(15, 5))

        s_db = dataframe.groupby('n_clusters')['davies_bouldin'].mean()

        plt.title('Davies-Bouldin score vs. K', fontsize=12)
        plt.plot([i for i in range(nclust_deb, nclust_fin)], s_db, linestyle="--", marker='o')

        plt.grid(False)
        plt.xlabel('K', fontsize=12)
        plt.ylabel('Davies-Bouldin score', fontsize=12)
        plt.xticks([i for i in range(nclust_deb, nclust_fin)], fontsize=11)
        plt.yticks(fontsize=11)

        plt.show()

    if cal_har:
        # Trace le graphique des indices de Calinski-Harabasz
        plt.figure(figsize=(15, 5))

        s_db = dataframe.groupby('n_clusters')['calinski_harabasz'].mean()

        plt.title('Calinski-Harabasz score vs. K', fontsize=12)
        plt.plot([i for i in range(nclust_deb, nclust_fin)], s_db,linestyle="--", marker='o')

        plt.grid(False)
        plt.xlabel('K', fontsize=12)
        plt.ylabel('Calinski-Harabasz score', fontsize=12)
        plt.xticks([i for i in range(nclust_deb, nclust_fin)], fontsize=11)
        plt.yticks(fontsize=11)

        plt.show()
        
        
        