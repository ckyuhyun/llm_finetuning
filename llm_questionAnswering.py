import collections

from datasets import load_dataset
import datasets
from datasets import concatenate_datasets
import pandas as pd
import os
import re
import numpy as np
from sklearn.metrics._scorer import metric
from tqdm.auto import tqdm
from transformers import (AutoTokenizer,
                          TrainingArguments,
                          Trainer,
                          AutoModelForQuestionAnswering, DefaultDataCollator, pipeline)


model_checkpoint = "distilbert/distilbert-base-uncased"


def prepare_answer_ds(dataset:pd.DataFrame):
    input = pd.DataFrame(columns=['context', 'question', 'answers'])
    for i in range(0, len(dataset)):
        answer = {}
        index = re.findall(r"([\d+]+)", dataset.iloc[i]['answer_position'])
        start_index = int(index[0])
        end_index = int(index[1])
        answer['text'] = [dataset.iloc[i]['context'][start_index:end_index]]
        answer['start_position'] = [start_index]
        answer['end_position'] = [start_index+end_index]

        input.loc[i] = {'context': dataset.iloc[i]['context'], 'question':dataset.iloc[i]['question'], 'answers':{'text': dataset.iloc[i]['context'][start_index:end_index], 'answer_start': start_index}}

    return input


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"]

        try:
            end_char = answer["answer_start"] + len(answer["text"])
        except:
            print(f'{i}')


        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

#
# def compute_metrics(start_logits, end_logits, features, examples):
#     example_to_features = collections.defaultdict(list)
#     for idx, feature in enumerate(features):
#         example_to_features[feature["example_id"]].append(idx)
#
#     predicted_answers = []
#     for example in tqdm(examples):
#         example_id = example["id"]
#         context = example["context"]
#         answers = []
#
#         # Loop through all features associated with that example
#         for feature_index in example_to_features[example_id]:
#             start_logit = start_logits[feature_index]
#             end_logit = end_logits[feature_index]
#             offsets = features[feature_index]["offset_mapping"]
#
#             start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
#             end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     # Skip answers that are not fully in the context
#                     if offsets[start_index] is None or offsets[end_index] is None:
#                         continue
#                     # Skip answers with a length that is either < 0 or > max_answer_length
#                     if (
#                         end_index < start_index
#                         or end_index - start_index + 1 > max_answer_length
#                     ):
#                         continue
#
#                     answer = {
#                         "text": context[offsets[start_index][0] : offsets[end_index][1]],
#                         "logit_score": start_logit[start_index] + end_logit[end_index],
#                     }
#                     answers.append(answer)
#
#         # Select the answer with the best score
#         if len(answers) > 0:
#             best_answer = max(answers, key=lambda x: x["logit_score"])
#             predicted_answers.append(
#                 {"id": example_id, "prediction_text": best_answer["text"]}
#             )
#         else:
#             predicted_answers.append({"id": example_id, "prediction_text": ""})
#
#     theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
#     return metric.compute(predictions=predicted_answers, references=theoretical_answers)


#model_checkpoint = "deepset/roberta-base-squad2"
dataset_path = "data_sets"
test_ds_path = os.path.join(os.path.curdir, dataset_path, "test_qa.csv")
train_ds_path = os.path.join(os.path.curdir, dataset_path, "train_qa.csv")


valid_columns = ['question','human_ans_indices','review','human_ans_spans']
rename_columns = ['question','human_ans_indices','context','answer']

if not os.path.isfile(test_ds_path) or not os.path.isfile(train_ds_path):
    raise Exception()

test_ds = pd.read_csv(test_ds_path)
train_ds = pd.read_csv(train_ds_path)
df_test = pd.DataFrame(test_ds, columns=valid_columns)
df_train = pd.DataFrame(train_ds, columns=valid_columns)

renaming_columns ={'question':'question','human_ans_indices':'answer_position','review':'context','human_ans_spans':'answers'}
df_test.rename(columns=renaming_columns, inplace=True)
df_train.rename(columns=renaming_columns, inplace=True)

# df_test['answers'] = df_test['human_ans_spans']
# df_train['answers'] = df_train['human_ans_spans']

df_test = prepare_answer_ds(df_test)
df_train = prepare_answer_ds(df_train)



val_dataset = datasets.Dataset.from_pandas(df_test)
train_dataset = datasets.Dataset.from_pandas(df_train)



# print(f'Columns : {[c+"," for c in df_test.columns]}')
# print(f'Question: {df_test.iloc[0].question}\nReview: {df_test.iloc[0].review}')

# Train
tokenzing_train_ds = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

# Validation
tokenzing_val_ds = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)


training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    #eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    #remove_unused_columns=True
)


model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
data_collator = DefaultDataCollator()


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= tokenzing_train_ds, #tokenized_squad["train"],
    eval_dataset= tokenzing_val_ds, # tokenized_squad["test"] ,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics
)


# predictions, _, _ = trainer.predict(tokenzing_val_ds)
# start_logits, end_logits = predictions
#compute_metrics(start_logits, end_logits, validation_dataset, val_dataset2)

trainer.train()

model_checkpoint2 = "my_awesome_qa_model"
model_path = os.path.join(os.path.curdir, "my_awesome_qa_model")
question_answerer = pipeline("question-answering")

question_answerer.save_pretrained(model_path)




context = train_ds.iloc[13].review
question = train_ds.iloc[13].question
result = question_answerer(question=question, context=context)


print('question : {0}'.format(question))
print('result : {0}'.format(result))







