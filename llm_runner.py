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


def get_latest_created_model():
    current_path = os.path.abspath(os.getcwd())
    sub_dir_path = trained_model_dic
    last_created_dir = None
    for dir_name in os.listdir(trained_model_dic):
        created_date = os.path.getctime(os.path.join(current_path, sub_dir_path, dir_name))
        if last_created_dir is None:
            last_created_dir = dir_name
        else:
            last_created_dir_date = os.path.getctime(os.path.join(current_path, sub_dir_path, last_created_dir))
            last_created_dir = last_created_dir if created_date < last_created_dir_date else dir_name

    return last_created_dir


#update_reference()
#generate_question_answers_dataset()


model = Training_Model(src_file_name="alertApprove.xlsx")
#model = Training_Model()
model.set(model_list.distilbert_base_uncased)

# Train configuration setting
model.set_tokenizer_configuration(
    do_eval=True,
    evaluation_strategy="epoch",
    learning_rate= 5e-5,
    num_train_epochs=2000)


# Training
model_dir_name = None
if bool(retrain_enable):
    model_dir_name = model.run(saving_trained_model=True,
                          evaluation_on=True)
else:
    model_dir_name = get_latest_created_model()

pipe = pipeline("question-answering",
                #model="distilbert/distilbert-base-uncased",
                token=model.get_tokenizing_ds(),
                model=os.path.join(trained_model_dic, model_dir_name)
                )


while True:
    # This will query for first user input, Name.

    question = str(input(Fore.RED+"\n\nMay I help you?"))

    context_collection = model.get_context_collection()
    tokenizer = model.get_tokenizer()
    loop_index = 0
    #print(f'The used model => {model_dir_name}\n')
    for context in context_collection:
        result = pipe(question=question, context=context)
        # Min value and Max value of score is 0 and 1.
        print(f'[{str(loop_index)}]answer : {result}')
        completed_answer = model.get_completed_answer(result.get('answer'))
        print(Fore.GREEN +f'[Completed answer]\n {completed_answer}')
        loop_index += 1

        print(f'Done........')






question = \
    "how to create an alert?"
    #"What is the best way to add records to a form?"
    #"How to open popup for reusable records?"
    #"What is the best way to add records to a form?"
    #"What is the process for creating alerts?"
    #"How to open popup for reusable records?"
    #"What is the purpose of alerts?"
    #"What if there are records that would be the same across libraries?"
    #"What is a product family?"
    #"how to get the reusable content alert?"

#context = "Alert can be submitted approval or rejection for updating of the data on the Alerts &  Approvals widget. Once opening a panel for a project on the Project Forms or a product family on the Reusable Content or Reusable Records tab page, there are a check button in green and a x button in red. The check button is for an approval, and the x button is for an rejection. if any button among the check button or x button got selected by clicking, the submit button would appear, and get highlighted the value that would be remained on the spot at the end. If the alert item in the Project Forms is submitted for an apporval, the project form page would be updated with the updated value that has been changed from the reusable content page or reusable records page by administrator. If the alert item in the Reusable Content is submitted for an approval, the all of project forms pages and the reusable records page that have the reusable content data would be updated with the updated value. . If the alert item in the Reusable Record is submitted for an approval, the all of project forms pages that have the reusable record data would be updated with the updated value. In addition, if the alert item for the reusable content data in the Reusable Records is submitted, it would ask if updating the reusable content data in the reusable content page on the Alert Submit modal. If you want to update the resuable content data as well along with updating for the reusable record data, make the checkbox checked and click the Confirm button on the Alert Submit modal. Once submitting the alert item, it would be removed from the widget"







