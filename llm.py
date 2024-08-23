import pandas as pd
import os
import re
import datasets
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer)

data_src_path = 'data_sets'

file_path = os.path.join(data_src_path, 'LLM_Materials.xlsx')

if not os.path.isfile(file_path):
    raise f'{file_path} is not existed'

model_path = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def update_answer_pos(dataSet: pd.DataFrame) -> pd.DataFrame:
    reference_context = None
    new_ds = pd.DataFrame(columns=dataSet.columns)

    for row_data in dataSet.iterrows():
        try:
            if reference_context is None:
                reference_context = row_data[1]['context']

            reference_context = row_data[1]['context'] if pd.notnull(row_data[1]['context']) and  reference_context != row_data[1]['context'] else reference_context
        except TypeError as e:
            raise e

        expect_answer = row_data[1]['answer'].strip()
        search_result = re.search(expect_answer, reference_context)  #reference_context.find(expect_answer)   #re.finditer(, row_data[1]['context'])

        try:
            start_pos = search_result.regs[0][0]
            end_pos = search_result.regs[0][1] #start_pos+len(row_data[1]["answer"])
            start_end_pos = f'{start_pos},{end_pos}'
            #ans_pos.append(start_end_pos)

            # new_row_data = { 'category': row_data[1]['category'],
            #                 'Title': row_data[1]['Title'],
            #                 'context': row_data[1]['context'],
            #                 'question': row_data[1]['question'],
            #                 'answer': row_data[1]['answer'],
            #                 'answer_pos': start_end_pos,}
            new_row_data = [row_data[1]['category'],
                            row_data[1]['Title'],
                            reference_context,
                            row_data[1]['question'],
                            row_data[1]['answer'],
                            start_end_pos]
            new_ds.loc[len(new_ds.index)] = new_row_data



        except AttributeError:
            print(f'answer reference : {row_data[1]["answer"]}')
            raise Exception('')
        try:
            print(f'extracted answer : {reference_context[start_pos: end_pos].strip()}')
        except Exception as e:
            raise Exception(f'{reference_context}')


    return new_ds

def cleaning_dataset(dataSet: pd.DataFrame) -> pd.DataFrame:
    dataSet.columns = dataSet.columns.str.replace(' ', '')
    filtered_dataset = dataSet.loc[~dataSet["question"].isna() & ~dataSet["answer"].isna() ]

    return filtered_dataset


def tokenizing(examples):
#def tokenizing(context, question, answer_pos):
    inputs = tokenizer(examples['question'],
                      examples['context'],
                      max_length=384,
                      padding=True,
                      truncation=True,
                      add_special_tokens=True,
                    return_offsets_mapping=True,
                      #is_split_into_words=True, # The sequence or batch of sequences to be encoded.
                                        # Each sequence can be a string or a list of strings (pretokenized string).
                                        # If the sequences are provided as list of strings (pretokenized),
                                        # you must set is_split_into_words=True (to lift the ambiguity with a batch of sequences).
                                )

    offset_mapping = inputs.pop('offset_mapping')
    start_pos = []
    end_pos = []
    for i, offset in enumerate(offset_mapping):
        answer_start_end_position = examples['answer_pos'][i]
        answer_start = int(re.split(',', answer_start_end_position)[0].strip())
        answer_end = int(re.split(',', answer_start_end_position)[1].strip())
        sequence_ids = inputs.sequence_ids(i)

        index=0
        while sequence_ids[index] != 1: #question token has '0'
            index += 1
        context_start = index

        while sequence_ids[index] == 1: #context token has '1'
            index += 1
        context_end = index - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > answer_end or offset[context_end][1] < answer_start :
            start_pos.append(0)
            end_pos.append(0)
        else:
            index = context_start
            while index < context_end and offset[index][0] <= answer_start:
                index += 1
            start_pos.append(index-1)

            while index >= context_start and offset[index][1] >= answer_end:
                index -= 1
            end_pos.append(index+1)



    inputs["start_positions"] = start_pos
    inputs["end_positions"] = end_pos

    return inputs

try:
    ds = pd.read_excel(file_path)
except pd.errors.EmptyDataError:
    raise f'{file_path} is empty'

if ds.empty:
    raise f'{file_path} has no data'

cleaned_dataset = cleaning_dataset(ds)
final_dataSet = update_answer_pos(cleaned_dataset)

valid_columms = ['context', 'question', 'answer', 'answer_pos']
#remaing_columms = ['question', 'start_pos', 'context', 'answer']
valid_ds = pd.DataFrame(final_dataSet, columns=valid_columms)

train_dataset = datasets.Dataset.from_pandas(valid_ds)

token_ds = train_dataset.map(tokenizing, batched=True, remove_columns=train_dataset.column_names)
#token_ds = tokenizing(train_dataset['question'],train_dataset['context'], train_dataset['answer_pos'])
model = AutoModelForQuestionAnswering.from_pretrained(model_path)



training_args = TrainingArguments(
    output_dir=f"fine_llm_result",
    #evaluation_strategy = "epoch",
    evaluation_strategy = "no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    do_eval=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= token_ds,
    #eval_dataset=tokenized_datasets["validation"].select(range(100)),
    #data_collator=data_collator,
    tokenizer=tokenizer,
    #train_dataset=small_train_dataset,
    #eval_dataset=small_eval_dataset,
    #compute_metrics=compute_metrics,
)

trainer.train()


pass



