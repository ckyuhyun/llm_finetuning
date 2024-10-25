import config_parser
from transformers import(
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
class distilbert_base_uncased_model:
    def __init__(self):
        self.check_point = "distilbert/distilbert-base-uncased"
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.check_point)
        self.__tokenizer__ = AutoTokenizer.from_pretrained(self.check_point)

