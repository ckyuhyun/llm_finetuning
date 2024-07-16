import pandas as pd
import os




data_src_path = 'qa_datasets'

file_path = os.path.join(data_src_path, 'test_qa.csv')

if not os.path.isfile(file_path):
    raise f'{file_path} is not existed'

ds = pd.read_csv(file_path)

if ds.empty:
    raise f'{file_path} has no data'

valid_columms = ['question', 'review']
valid_ds = pd.DataFrame(ds, columns=valid_columms)







