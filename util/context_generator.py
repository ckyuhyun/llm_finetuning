import os
import re
import docx2txt
import pandas as pd

category_list = None
from transformers import(
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


access_token = "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"

def check_get_category(content) -> (bool, str):
    _content = re.sub(r'\n', '', re.sub(r'\t','', re.sub(r'\t\d+', '', content)))
    for c in category_list:
        if _content == c:
            return True, _content
    return False, ''


def update_category(contents:list):
    global  category_list
    _list = [re.sub(r'\t\d+', '', c) for c in contents if re.search('\t\d+', c) != None]
    list = [re.sub(r'\t', '', i) for i in _list]
    list.append('Contents')
    category_list = list





def update_reference():
    current_category = None

    my_text = docx2txt.process(os.path.join("data_sets", "Training Notes.docx"))

    contents = my_text.split("\n\n")
    context = {}
    category_content = ''
    start_new_key = False
    start_content_update = False
    ignore_content = True
    previous_category = None
    answers_candidates = []

    update_category(contents)

    for c in contents:
        isCategory, category = check_get_category(c)
        if isCategory:
            if category not in context.keys() and current_category in context.keys():
                context[current_category] = category_content
                category_content = ''
            current_category = category
        else:
            if current_category not in context.keys():
                context[current_category] = ''
                category_content = f'{c}\n'
            else:
                category_content += f'{c}\n'


    df = pd.DataFrame.from_dict(context, orient='index')
    df.to_csv('context_file.csv')


def generate_question_answers_dataset():
    #check_point= 'google/flan-t5-large'
    #check_point = 'TheBloke/Llama-2-7B-GPTQ'
    check_point = "iarfmoose/t5-base-question-generator"
    #check_point= "voidful/context-only-question-generator"
    #model = T5ForConditionalGeneration.from_pretrained(check_point, token=access_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(check_point, token=access_token)
    #tokenizer = T5Tokenizer.from_pretrained(check_point, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(check_point, token=access_token, use_fast=False)

    reference_doc_path = os.path.join('data_sets', 'alertApprove.xlsx')
    context_data = pd.read_excel(reference_doc_path)
    #context_ds = pd.DataFrame(context_data, columns=["category", "context"])
    #context = context_data.iloc[5,1]


    #context = "Highlight the menu. Parts and Projects where users will spend their time. Wrench icon to access the Admin section. Point out the page footer is available on every page. The footer has support contact information, legal, etc. Knowledge base to find additional information. Videos and FAQ. Open the site, very quickly highlight there is lots of help available, point out Module A and B training videos."

    #answer = "Product families are a collection of common form records made up of Operations, Product Characteristics, and Process Characteristics required to produce the product."
    answer_list = list(context_data.iloc[1:,-1].values)
    context = context_data.iloc[1,1]

    #input_text = f"generate question: context: {context}"

    for answer in answer_list:
        input_text = f"<answer> {answer} <context> {context}"

        input_ids = tokenizer.encode(input_text,
                                     return_tensors='pt',
                                     max_length=512,
                                     padding="max_length",
                                     truncation="only_second")
                                     #num_return_sequences=5

        outputs = model.generate(input_ids, max_length=512)
        questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        print(f'Answer: {answer}\n\nQuestion list:\n{questions}')

    pass








