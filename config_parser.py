import configparser

def __get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def huggingface_token_acess():
    return __get_config()['Huggingface']['AccessToken']




