def test_pk(df, columns):
    if len(df) == len(df[columns].drop_duplicates()):
        print("La clé est unique")
    else:
        print("La clé n'est pas unique \n")
        print(f'Les valeurs dupliquées dans {columns} sont : \n')
        print(df[df[columns].duplicated()], '\n')
        print(f'Le nombre de valeurs dupliquées sur {len(df[:])} est de : \n')
        print(df[df[columns].duplicated()].count(), '\n')
        print('Essaye une autre colonne \n')
  