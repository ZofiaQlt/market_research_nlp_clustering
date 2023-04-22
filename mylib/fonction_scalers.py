from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler  
from sklearn.preprocessing import QuantileTransformer
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

bold = "\033[1m"
red = "\033[31m"
end = "\033[0;0m"


def scalers(df):
    # on détermine les colonnes numériques
    num = []
    for i in df.columns:
        if df[i].dtypes == int or df[i].dtypes == float:
            num.append(i)
    
    print(bold + "OUTLIERS" + end)
    print("-" * 100)
    # on calcule le z-score sur toutes les variables numériques
    for i in num:
        df.sort_values(by=i, inplace=True)
        outlier_z = df[stats.zscore(df[i]) > 1.96]
        nb_outlier_z = len(outlier_z)
        print(bold + f"\n{i} : {nb_outlier_z} outliers \n\n" + end + f"{outlier_z[['Pays', i]]} \n" + "-" * 50)
    
    #création de DF avec valeurs numériques uniquement    
    df_scal = df.copy()
    df_scal = df[num]
    #print(df_scal.sample())
    
    print(bold + "DISTRIBUTIONS AVANT SCALING"+ end)
    print("-" * 100)
    # check data before scaling
    data = df_scal.values
    # convert the array back to a dataframe
    dataset = pd.DataFrame(data)
    # summarize
    print(dataset.describe())
    # histograms of the variables
    dataset.hist()
    pyplot.show()
    print("-" * 100)
    
    print(bold + "SCALERS" + end)
    print("-" * 100)
    scalers = [PowerTransformer(method='yeo-johnson', standardize=True), RobustScaler(), StandardScaler(), MinMaxScaler(), MaxAbsScaler(),                             QuantileTransformer(output_distribution='normal')]
    # check data after scaling
    data = df_scal.values
    # perform a robust scaler transform of the dataset
    for i in scalers:
        trans = i
        data = trans.fit_transform(data)
        # convert the array back to a dataframe
        df_scal = pd.DataFrame(data)
        # summarize
        print(bold + str(i) + end)
        print(df_scal.describe())
        # histograms of the variables
        df_scal.hist()
        pyplot.show()
        
        # création boxplots
        fig=plt.figure(figsize=(25,15))
        df_scal.boxplot(vert=False)
        plt.yticks(size=20)
        plt.show()
        print("-" * 100)
    
    