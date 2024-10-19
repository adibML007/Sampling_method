import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import dask.dataframe as dd

ddf = dd.read_csv(r"D:/mimic-iv-readm.csv", header=None, blocksize="400mb")
print(ddf)
