import config_parser
from transformers import(
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


class T5_base_question_generate_model:
    def __init__(self):
        self.check_point = "iarfmoose/t5-base-question-generator"
        self.model =AutoModelForSeq2SeqLM.from_pretrained(self.check_point, token=config_parser.huggingface_token_acess),
        self.tokenizer = AutoTokenizer.from_pretrained(self.check_point, token=config_parser.huggingface_token_acess, use_fast=False)