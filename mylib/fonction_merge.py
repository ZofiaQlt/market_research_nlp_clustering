def df_merge(df1, df2, column):
    df = df1.merge(df2, on=column, how='outer', indicator=True)
    df_both = df[df['_merge'] == 'both']
    df_left = df[df['_merge'] == 'left_only']
    df_right = df[df['_merge'] == 'right_only']
    proportion_both = len(df_both) / len(df) * 100
    proportion_left = len(df_left) / len(df) * 100
    proportion_right = len(df_right) / len(df) * 100
    print("\033[31m" + f"Le DataFrame contient {len(df)} lignes")
    print("\033[31m" + f"\nLa proportion de 'both' est de {round(float(proportion_both), 3)} % soit {len(df_both)} lignes")
    print("\033[31m" + f"La proportion de 'left_only' est de {round(float(proportion_left), 3)} % {len(df_left)} lignes")
    print("\033[31m" + f"La proportion de 'right_only' est de {round(float(proportion_right), 3)} % {len(df_right)} lignes")
    if proportion_both < 100:
        print("\033[31m" + f"\nAperçu des données selon les jointures 'both', 'left_only' ou 'right_only' :" + "\033[0;0m")
        return df_both.head(), df_left.head(), df_right.head()
    
    
    


    