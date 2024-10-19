import dask.dataframe as dd

df = dd.read_csv('D:/mimic-iv-readm.csv', header=None, include_path_column=False)
print(df.shape)
