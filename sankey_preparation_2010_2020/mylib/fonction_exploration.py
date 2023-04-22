import pandas as pd
from IPython.display import display_html 

def exploration1(df):

    a = pd.DataFrame(df.head(3))
    b = pd.DataFrame(df.tail(3))
    c = pd.DataFrame(df.sample(3))
    df1_styler = a.style.set_table_attributes("style='display:grid'").set_caption('__________ Head __________')
    df2_styler = b.style.set_table_attributes("style='display:grid'").set_caption('__________ Tail __________')
    df3_styler = c.style.set_table_attributes("style='display:grid'").set_caption('__________ Sample __________')
    space = "\xa0" * 1
    display_html(df1_styler._repr_html_() + space + df2_styler._repr_html_() + space + df3_styler._repr_html_(), raw=True)
    
    
def exploration2(df):
    df_info = pd.DataFrame(df.count()).T.rename(index={0:"Nombre de valeurs totales"})
    df_info = df_info.append(pd.DataFrame(df.dtypes).T.rename(index={0:"Type des données"}))
    df_info = df_info.append(pd.DataFrame(df.isna().sum()).T.rename(index={0:"Nombre de NaN"}))
    df_info = df_info.append(pd.DataFrame(df.isna().sum()/df.shape[0]*100).T.rename(index={0:"NaN en %"}))
    df_info = df_info.append(pd.DataFrame(df.nunique()).T.rename(index={0:"Nombre de valeurs uniques"}))
    df_info = df_info.append(pd.DataFrame(df.nunique()/df.shape[0]*100).T.rename(index={0:"Valeurs uniques en%"}))
    df_info = df_info.T
    return df_info

    
def exploration3(df):
    print()
    print('---------------------------------------')
    print('Nombre de lignes et de colonnes (shape)')
    print('--------------------------------------- \n')
    print(df.shape, '\n')
    print('--------------------------------------')
    print('Affichage des NaN (isna().any(axis=1))')
    print('-------------------------------------- \n')
    print(df[df.isna().any(axis=1)], '\n')
    print('---------------------------------------')
    print('Nombre de doublons (duplicated().sum())')
    print('--------------------------------------- \n')
    print(df.duplicated().sum(), '\n')
    print('-----------------------------------------------------')
    print('Affichage des doublons (df[df.duplicated()].head(10))')
    print('----------------------------------------------------- \n')
    print(df[df.duplicated()].head(10), '\n')  
    

def exploration4(df):
    return pd.DataFrame(df.describe(include='all').T)

    
def exploration_long(df):
    print()
    print('--------------------------')
    print('Aperçu du DataFrame (head)')
    print('-------------------------- \n')
    print(df.head(), '\n')
    print('------------------------------------')
    print('Aperçu de la fin du DataFrame (tail)')
    print('------------------------------------ \n')
    print(df.tail(), '\n')
    print('---------------------------------')
    print('Échantillon du DataFrame (sample)')
    print('--------------------------------- \n')
    print(df.sample(5), '\n')
    print('---------------------------------------')
    print('Nombre de lignes et de colonnes (shape)')
    print('--------------------------------------- \n')
    print(df.shape, '\n')
    print('-------------------------')
    print('Type des données (dtypes)')
    print('------------------------- \n')
    print(df.dtypes, '\n')
    print('------------')
    print('Infos (info)')
    print('------------ \n')
    print(df.info(), '\n')
    print('----------------------------')
    print('Nombre de NaN (isna().sum())')
    print('---------------------------- \n')
    print(df.isna().sum(), '\n')
    print('--------------------------------------')
    print('Affichage des Nan (isna().any(axis=1))')
    print('-------------------------------------- \n')
    print(df[df.isna().any(axis=1)], '\n')
    print('---------------------------------------')
    print('Nombre de doublons (duplicated().sum())')
    print('--------------------------------------- \n')
    print(df.duplicated().sum(), '\n')
    print('-----------------------------------------------------')
    print('Affichage des doublons (df[df.duplicated()].head(10))')
    print('----------------------------------------------------- \n')
    print(df[df.duplicated()].head(10), '\n')      
    print('-----------------------------------')
    print('Résumé analyse univariée (describe)')
    print('----------------------------------- \n')
    print(df.describe(include='all'))
