import os
import requests
import json
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    AdamW,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DistilBertForQuestionAnswering)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from datasets import(
    load_dataset,
    disable_progress_bar,
    Dataset
)

import random


from huggingface_hub.hf_api import HfFolder
#HfFolder.save_token()


# if not os.path.exists('squard'):
#     os.makedirs('squard')
#
# url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
# res = requests.get(f'{url}train-v2.0.json')

def read_squad(path):
    with open(os.path.join(os.getcwd(), path)) as f:
        data = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']

                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'

                for answer in qa[access]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        if gold_text in context:
            answer['answer_end'] = end_idx if context[start_idx:end_idx] is gold_text else context.find(gold_text) + len(gold_text)
        else:
            raise Exception(f'this context does not have {gold_text}')


# def add_token_position(encodings, answers):
#     start_positions = []
#     end_positions = []
#
#     for i in range(len(answers)):
#         start_positions.append(encodings.ch)


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        shift = 1

        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def get_config_bitsandbytes():
    compute_type = getattr(torch, "float16")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_type,
        bnb_4bit_use_double_quant=False
    )

cache_dir = None


train_contexts, train_questions, train_answers = read_squad('squad\\train-v2.0.json')
datasets1 = load_dataset("json", data_ files='squad\\dev-v2.0.json')
datasets2 = load_dataset("squad_v2")
val_contexts, val_questions, val_answers = read_squad('squad\\dev-v2.0.json')

print(datasets2)
#add_end_idx(train_answers, train_contexts)
#add_end_idx(val_answers, val_contexts)


model_path = 'distilbert-base-uncased'
# Tokenizer
#tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


tokenizer = DistilBertTokenizerFast.from_pretrained(model_path,
                                                    #use_auth_token=True,
                                                    token='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')

partial_train_question =train_questions # random.sample(train_questions, 100)
partial_train_context = train_contexts # random.sample(train_contexts, 100)


train_encodings = tokenizer(partial_train_question,
                            partial_train_context,
                            #truncation="only_second",
                            truncation=True,
                            padding=True,
                            return_offsets_mapping=True,
                            return_overflowing_tokens=True,
                            #max_length=384
                            )




partial_evl_context = val_contexts # random.sample(val_contexts,100)
partial_evl_question = val_questions # random.sample(val_questions,100)
eval_encodings = tokenizer(partial_evl_context,
                           partial_evl_question,
                           #truncation="only_second",
                           truncation=True,
                           padding=True,
                           return_offsets_mapping=True,
                           return_overflowing_tokens=True
                           )



# We will label impossible answers with CLS token's index.
cls_index = 0




def prepare_qa_train(example):
    sample_mapping = example['overflow_to_sample_mapping']  # contains some tokens from the end of the truncated sequence
    offset_mapping = example['offset_mapping']  # offset of each token

    # start_positions and end_positions will be the labels for extractive question answering
    example["start_positions"] = []
    example["end_positions"] = []


    for i, offset in enumerate(offset_mapping):
        input_ids = example['input_ids'][i] # input_ids contains ids of all tokens for each slide

        sample_index = sample_mapping[i]
        answer = train_answers[sample_index]

        if answer["answer_start"] == 0:
            example["start_positions"].append(cls_index)
            example["end_positions"].append(cls_index)
        else:
            start_char = answer["answer_start"]
            end_char = start_char + len(answer["text"])

            sequence_ids = example.sequence_ids(i) # lists indicating the sequence ids corresponding to each token

            # find the context's corresponding start and end token index
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # if answer is within the context offset, move the token_start_index and token_end_index
            # to two ends of the answer else label it with cls index
            offset_start_char = offset[token_start_index][0]
            offset_end_char = offset[token_end_index][1]

            if offset_start_char <= start_char and offset_end_char >= end_char:
                while token_start_index < len(offset) and offset[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_position = token_start_index - 1

                while offset[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_position = token_end_index + 1

                example["start_positions"].append(start_position)
                example["end_positions"].append(end_position)
            else:
                example["start_positions"].append(cls_index)
                example["end_positions"].append(cls_index)

    return example

'''
start_positions = train_encodings["start_positions"]
end_positions = train_encodings["end_positions"]

for i, input_ids in enumerate(train_encodings["input_ids"]):
    start = start_positions[i]
    end = end_positions[i] +1
    string = tokenizer.decode(input_ids[start:end])
    expect_answer = train_answers[0].get('text')

'''

torch.set_num_threads(1)

#tokenized_datasets = prepare_qa_train(train_encodings)
#tokenized_evl_datasets = prepare_qa_train(eval_encodings)
tokenized_datasets = datasets2.map(
                            prepare_qa_train,
                            #with_indices=True,
                            batched=True,  # actually let you control the size of the generated dataset freely.
                            #remove_columns=datasets2["train"].column_names,
                            #num_proc=4  # allows you to set the number of processes to use.
                    )

#tokenized_datasets.save_to_disk('.')

#print(tokenized_datasets)



#val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
#add_token_positions(train_encodings, train_answers)


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
#val_dataset = SquadDataset(val_encodings)


###############################################
###############################################
###############################################



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



device_map={"": 0}
model = DistilBertForQuestionAnswering.from_pretrained(model_path,
                                                       #device_map=device_map,
                                                       #quantization_config=get_config_bitsandbytes(),
                                                       #trust_remote_code=True,
                                                       #use_auth_token=True,
                                                       token='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'
                                                       )
'''
model = DistilBertForQuestionAnswering.from_pretrained(model_path)
'''
from transformers import pipeline
'''
pl = pipeline('question-answering', model=model, tokenizer=tokenizer)

#for index in range(len(val_questions)):
for index in range(10):
    output = pl({
        "question": val_questions[index],
        "context": val_contexts[0]
    })
    print('output answer : {0}, validation answer: {1}'.format(output.get('answer'), val_answers[index]["text"]))


'''

answer = val_answers[0]["text"]

hit_count = 0
false_count = 0
#for question in val_questions:




#answer = model(question)["answer"]

print(answer)

import numpy as np
import evaluate


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer",
                                  logging_steps=1,
                                  per_device_train_batch_size=1,
                                  num_train_epochs=1,
                                  evaluation_strategy="steps",
                                  #remove_unused_columns=False,
                                  eval_steps=5,
                                  no_cuda=True
                                  )




print('train encoding key:')
for k in list(train_encodings.keys()):
    print(f'{k}, ')

print('eval encoding key:')
for k in list(eval_encodings.keys()):
    print(f'{k}, ')


model.to(device)
trainer = Trainer(
    model=model,
    #args=training_args,
    train_dataset=train_encodings,
    #eval_dataset=eval_encodings,
    #tokenizer=tokenizer
    #compute_metrics=compute_metrics\
)

# move model over to detected device
#trainer.to(device)

trainer.train()
#optim = AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(1):
    trainer.train()
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
#        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        # train model on batch and return outputs (incl. loss)
        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)



        # extract loss
        loss = outputs[0]
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())



# model_path = 'models/distilbert-custom'
# model = DistilBertForQuestionAnswering.from_pretrained(model_path)
# tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)


#print(train_answers[0])
#print('\n\n')
#print(tokenizer.decode(train_encodings['input_ids'][0]))




# print(f'{train_contexts[0]}\n')
# print(f'{train_questions[0]}\n')
# print(f'{train_answers[0]}\n')




