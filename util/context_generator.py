import os
import re
import docx2txt
import pandas as pd

category_list = None

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


