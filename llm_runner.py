import os.path
from transformers import pipeline
from llm import Data_Preprocessing, Training_Model
# from ray.train.torch import TorchTrainer
# from ray.train import ScalingConfig

# def train_func():
#     pass
#
# scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
# trainer = TorchTrainer(train_func, scaling_config=scaling_config)
# result = trainer.fit()


model = Training_Model()

# Train configuration setting
model.set_tokenizer_configuration(
    do_eval=True,
    num_train_epochs=190)


# Training
model.run(
    saving_trained_model=False,
          evaluation_on=True)

trained_model_dic = "trained_model_dic"
model_name = '2024_8_2518_12_llm_model'
pipe = pipeline("question-answering",
                model=os.path.join(trained_model_dic, model_name))

question = \
    "How could find out how many alerts exist?"
    #"how to get the reusable content alert?"

#context = "Alert can be submitted approval or rejection for updating of the data on the Alerts &  Approvals widget. Once opening a panel for a project on the Project Forms or a product family on the Reusable Content or Reusable Records tab page, there are a check button in green and a x button in red. The check button is for an approval, and the x button is for an rejection. if any button among the check button or x button got selected by clicking, the submit button would appear, and get highlighted the value that would be remained on the spot at the end. If the alert item in the Project Forms is submitted for an apporval, the project form page would be updated with the updated value that has been changed from the reusable content page or reusable records page by administrator. If the alert item in the Reusable Content is submitted for an approval, the all of project forms pages and the reusable records page that have the reusable content data would be updated with the updated value. . If the alert item in the Reusable Record is submitted for an approval, the all of project forms pages that have the reusable record data would be updated with the updated value. In addition, if the alert item for the reusable content data in the Reusable Records is submitted, it would ask if updating the reusable content data in the reusable content page on the Alert Submit modal. If you want to update the resuable content data as well along with updating for the reusable record data, make the checkbox checked and click the Confirm button on the Alert Submit modal. Once submitting the alert item, it would be removed from the widget"


context_collection = model.get_context_collection()
tokenizer = model.get_tokenizer()
loop_index = 0
for context in context_collection:
    result = pipe(question=question, context=context, tokenizer=tokenizer)
    # Min value and Max value of score is 0 and 1.
    print(f'[{str(loop_index)}]answer : {result}')
    loop_index += 1


