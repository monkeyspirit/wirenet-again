import pandas as pd


def _get_dataset():
    excel = pd.read_excel('../rawSource/EmberFeatures.xlsx')
    excel = excel[['Numero colonna', 'Funzione']]
    return excel[excel['Numero colonna'].notnull()]


def _expand_df(dataframe):
    to_append = []
    for _, col_index, value in dataframe.itertuples():
        try:
            index = int(col_index)
            to_append.append([index, value])
        except ValueError:
            col_index = col_index.split('-')
            for i in range(int(col_index[0]), int(col_index[1])+1):
                to_append.append([i, value])

    return pd.DataFrame(to_append, columns=['Numero colonna', 'Funzione'])


def get_name(index: str):
    """
    Returns all names of field from index in format 'Column_####'
    """
    index = int(index[7:])
    return dataset.loc[dataset['Numero colonna'] == index, 'Funzione'].values


dataset = _expand_df(_get_dataset())
