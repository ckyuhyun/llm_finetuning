import pandas as pd
import os




data_src_path = 'qa_datasets'

file_path = os.path.join(data_src_path, 'test_qa.csv')

if not os.path.isfile(file_path):
    raise f'{file_path} is not existed'

ds = pd.read_csv(file_path)

if ds.empty:
    raise f'{file_path} has no data'

valid_columms = ['question','human_ans_indices','review','human_ans_spans']
remaing_columms = ['question','start_pos','context','answer']
valid_ds = pd.DataFrame(ds, columns=valid_columms)



def ds_pre_processing(dataset:pd.DataFrame):
    input = pd.DataFrame(columns=remaing_columms)
    for i in range(0, len(dataset.values)):
        input[i]['question'] = dataset.values[i]['question']

    pass


ds_pre_processing(pd.DataFrame(valid_ds, columns=remaing_columms))













