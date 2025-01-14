import os.path
from colorama import Fore, Back, Style
from transformers import pipeline
from llm import Data_Preprocessing, Training_Model
from model.model_list import model_list
from util.context_generator import update_reference, generate_question_answers_dataset
# from ray.train.torch import TorchTrainer
# from ray.train import ScalingConfig

# def train_func():
#     pass
#
# scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
# trainer = TorchTrainer(train_func, scaling_config=scaling_config)
# result = trainer.fit()


retrain_enable = True
trained_model_dic = "trained_model_dic"
token_ds_dic='token_ds_dic'

def get_latest_created_model(folder_name:str):
    current_path = os.path.abspath(os.getcwd())
    sub_dir_path = folder_name
    last_created_dir = None
    for dir_name in os.listdir(folder_name):
        created_date = os.path.getctime(os.path.join(current_path, sub_dir_path, dir_name))
        if last_created_dir is None:
            last_created_dir = dir_name
        else:
            last_created_dir_date = os.path.getctime(os.path.join(current_path, sub_dir_path, last_created_dir))
            last_created_dir = last_created_dir if created_date < last_created_dir_date else dir_name

    return last_created_dir



#update_reference()
#generate_question_answers_dataset()


model = Training_Model(src_file_name="alertApprove.xlsx",
                       trained_model_dic=trained_model_dic,
                       token_ds_dic=token_ds_dic)

model.set(model_list.distilbert_base_uncased)


# Train configuration setting
model.set_tokenizer_configuration(
    do_eval=True,
    evaluation_strategy="epoch",
    learning_rate= 5e-5,
    num_train_epochs=10)


# Training
model_dir_name = None

#model.run_ray_tune()
if bool(retrain_enable):
    model_dir_name = model.run(saving_trained_model=True,
                               evaluation_on=True)
else:
    model_dir_name = get_latest_created_model(trained_model_dic)



token_ds = model.get_tokenizing_ds()
pipe = pipeline("question-answering",
                #model="distilbert/distilbert-base-uncased",
                token= token_ds,
                model=os.path.join(trained_model_dic, model_dir_name)
                )


while True:
    # This will query for first user input, Name.

    question = str(input(Fore.RED+"\n\nMay I help you?"))

    if question: # check if questioned
        context_collection = model.get_context_collection()
        tokenizer = model.get_tokenizer()

        #print(f'The used model => {model_dir_name}\n')
        for context in context_collection:
            result = pipe(question=question, context=context)
            # Min value and Max value of score is 0 and 1.
            '''
            Score : The ‘score’ field represents the confidence score of the predicted answer, with a value between 0 and 1. 
                    A higher score indicates a higher level of confidence in the answer. In this case, the score is 0.99887, 
                    which is close to 1, indicating a high level of confidence in the answer.
            '''
            print(f'Answer : {result}')

            rounded_score = round(result.get('score'), 2)
            if rounded_score > 0.8: # Target of probability
                completed_answer = model.get_completed_answer(result.get('answer'))

                if completed_answer is not None:
                    print(Fore.GREEN + f'[Completed answer]\n {completed_answer}')
                else:
                    print(Fore.GREEN + f'[Completed answer]\n Can not find an appropriate answer')
            else:
                print(Fore.GREEN + f'[Completed answer]\n Can not find an appropriate answer')

            print(f'Done........')


#question = \
    #"how to create an alert?"
    #"What is the best way to add records to a form?"
    #"How to open popup for reusable records?"
    #"What is the best way to add records to a form?"
    #"What is the process for creating alerts?"
    #"How to open popup for reusable records?"
    #"What is the purpose of alerts?"
    #"What if there are records that would be the same across libraries?"
    #"What is a product family?"
    #"how to get the reusable content alert?"









