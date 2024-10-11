from typing import Optional
import pandas as pd
import os
import re
import datasets
import evaluate
import numpy as np
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer)

from sklearn.metrics import f1_score
from model.distilbert_base_uncased import distilbert_base_uncased_model
from model.t5_base_question_generator import T5_base_question_generate_model
from model.model_list import model_list


class Data_Preprocessing:
    def __init__(self,
                 data_src_path,
                 file_path,
                 trained_model_dic):
        self.data_src_path = data_src_path
        self.file_path = file_path
        self.trained_model_dic = trained_model_dic
        self.preprocessed_ds = None
        self.__data_preprocessing()

    def get_preprocess_ds(self):
        return self.preprocessed_ds

    def __data_preprocessing(self):
        try:
            ds = pd.read_excel(self.file_path)
        except pd.errors.EmptyDataError:
            raise f'{self.file_path} is empty'

        if ds.empty:
            raise f'{self.file_path} has no data'

        self.preprocessed_ds = self.__cleaning_dataset(ds)

    def __cleaning_dataset(self, dataSet: pd.DataFrame) -> pd.DataFrame:
        dataSet.columns = dataSet.columns.str.replace(' ', '')
        filtered_dataset = dataSet.loc[~dataSet["question"].isna() & ~dataSet["answer"].isna()]

        return filtered_dataset


class Training_Model(distilbert_base_uncased_model,
                     T5_base_question_generate_model):
    def __init__(self,
                 data_src_path='data_sets',
                 src_file_name='LLM_Materials.xlsx',
                 trained_model_dic='trained_model_dic'):
        self.data_src_path = data_src_path
        self.file_path = os.path.join(self.data_src_path, src_file_name)
        self.trained_model_dic = trained_model_dic
        # self.check_point = check_point
        # self.tokenizer = AutoTokenizer.from_pretrained(self.check_point)
        self.training_args = None
        self.train_dataset = None
        self.valid_columms = ['context', 'question', 'answer', 'answer_pos']
        self.model = None
        self.checkpoint = None
        self.tokenizer = None
        self.token_ds = None
        # self.metric = evaluate.load("accuracy")

        self.data_Preprocessing = Data_Preprocessing(self.data_src_path, self.file_path, self.trained_model_dic)

        if not os.path.isfile(self.file_path):
            raise f'{self.file_path} is not existed'

    def set(self, model: model_list):
        if model is model_list.distilbert_base_uncased:
            model = distilbert_base_uncased_model()
        elif model is model_list.T5_base_question_generate:
            model = T5_base_question_generate_model()

        self.model = model.model
        self.checkpoint = model.check_point
        self.tokenizer = model.tokenizer

        final_dataSet = self.__update_answer_pos()
        valid_ds = pd.DataFrame(final_dataSet, columns=self.valid_columms)

        self.train_dataset = datasets.Dataset.from_pandas(valid_ds)

    def get_completed_answer(self, hint_answer):
        return [answer for answer in self.train_dataset['answer'] if hint_answer in answer][0]


    def get_tokenizer(self):
        return self.tokenizer

    def get_context_collection(self):
        contexts = list(dict.fromkeys(self.train_dataset['context']))
        return contexts

    def get_tokenizing_ds(self):
        return self.token_ds

    def run(self, saving_trained_model=False, evaluation_on=False) \
            -> Optional[str]:
        self.token_ds = self.train_dataset.map(self.tokenizing, batched=True, remove_columns=self.train_dataset.column_names)
        # token_ds = tokenizing(train_dataset['question'],train_dataset['context'], train_dataset['answer_pos'])
        # model = AutoModelForQuestionAnswering.from_pretrained(self.check_point)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.token_ds,
            # data_collator=data_collator,
            tokenizer=self.tokenizer,
            # train_dataset=small_train_dataset,
            # eval_dataset=small_eval_dataset,
            # compute_metrics=self.compute_metrics,
        )
        if evaluation_on:
            trainer.eval_dataset = self.token_ds

        trainer.train()

        if evaluation_on:
            results = trainer.evaluate()
            print(f'result : {results}')

        if saving_trained_model:
            if not os.path.isdir(self.trained_model_dic):
                os.mkdir(f'{self.trained_model_dic}')
            file_name = f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}{datetime.now().hour}_{datetime.now().minute}_llm_model"
            trainer.save_model(os.path.join(self.trained_model_dic, file_name))

        return file_name if saving_trained_model else None

    def set_tokenizer_configuration(self,
                                    output_dir=f"fine_llm_result",
                                    evaluation_strategy="steps",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=8,
                                    per_device_eval_batch_size=8,
                                    num_train_epochs=2,
                                    weight_decay=0.01,
                                    do_eval=False):
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            # evaluation_strategy
            # "no" - No evaluation is done during training
            # "steps" - Evaluation is done every each step
            # "epoch" - Evaluation is done every epoch
            evaluation_strategy=evaluation_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            do_eval=do_eval
        )

    # def compute_metrics(self, pred):
    # logits, labels = pred
    # prediction = np.argmax(logits, axis=-1)
    # return self.metric(predictions=prediction, references=labels)
    # squad_labels = pred.label_ids
    # squad_preds = pred.predictions.argmax(-1)
    #
    # # Calculate Exact Match (EM)
    # em = sum([1 if p == l else 0 for p, l in zip(squad_preds, squad_labels)]) / len(squad_labels)
    #
    # Calculate F1-score
    # f1 = f1_score(squad_labels, squad_preds, average='macro')
    #
    # return {
    #     'exact_match': em,
    #     'f1': f1
    # }

    def __update_answer_pos(self) -> pd.DataFrame:
        dataSet = self.data_Preprocessing.get_preprocess_ds()
        reference_context = None
        new_ds = pd.DataFrame(columns=dataSet.columns)

        for row_data in dataSet.iterrows():
            try:
                if reference_context is None:
                    reference_context = row_data[1]['context']

                reference_context = row_data[1]['context'] if pd.notnull(row_data[1]['context']) and reference_context != row_data[1]['context'] else reference_context
            except TypeError as e:
                raise e

            expect_answer = row_data[1]['answer'].strip()
            search_result = re.search(expect_answer, reference_context)  # reference_context.find(expect_answer)   #re.finditer(, row_data[1]['context'])

            try:
                if search_result is not None:
                    start_pos = search_result.regs[0][0]
                    end_pos = search_result.regs[0][1]  # start_pos+len(row_data[1]["answer"])
                    start_end_pos = f'{start_pos},{end_pos}'
                    # ans_pos.append(start_end_pos)

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

                    try:
                        new_ds.loc[len(new_ds.index)] = new_row_data
                    except ValueError as ve:
                        print(ve)

            except AttributeError as e:
                print(f'answer reference : {row_data[1]["answer"]}')
                raise Exception(f'{e.message}')
            # try:
            #     print(f'extracted answer : {reference_context[start_pos: end_pos].strip()}')
            # except Exception as e:
            #     raise Exception(f'{reference_context}')

        return new_ds

    def tokenizing(self, examples):
        inputs = self.tokenizer(examples['question'],
                                examples['context'],
                                max_length=384,
                                padding=True,
                                truncation=True,
                                add_special_tokens=True,
                                return_offsets_mapping=True,
                                # is_split_into_words=True, # The sequence or batch of sequences to be encoded.
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

            index = 0
            while sequence_ids[index] != 1:  # question token has '0'
                index += 1
            context_start = index

            while sequence_ids[index] == 1:  # context token has '1'
                index += 1
            context_end = index - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > answer_end or offset[context_end][1] < answer_start:
                start_pos.append(0)
                end_pos.append(0)
            else:
                index = context_start
                while index < context_end and offset[index][0] <= answer_start:
                    index += 1
                start_pos.append(index - 1)

                while index >= context_start and offset[index][1] >= answer_end:
                    index -= 1
                end_pos.append(index + 1)

        inputs["start_positions"] = start_pos
        inputs["end_positions"] = end_pos

        return inputs
