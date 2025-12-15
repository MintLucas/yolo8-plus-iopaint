import sys,os
sys.path.append(os.getcwd())
from util.mylogging import get_logger


class Model_config:
    def __init__(self, log=get_logger("model_config")):
        self.log = log
        self.prompt_default = self.get_prompt("config/prompt/cat_rec_star/create_blog_v4_mode0.md")
        self.config_dict = {}
        self.init_all_prompt()
    def init_all_prompt(self):

        self.prompt_star = self.get_prompt("config/prompt/cat_rec_star/create_blog_v4_mode0.md")
        self.prompt_life = self.get_prompt("config/prompt/cat_rec_star/life_create_blog_v1_mode0.md")
        self.prompt_live = self.get_prompt("config/prompt/cat_rec_star/live_create_blog_v1_mode0.md")
        self.prompt_car = self.get_prompt("config/prompt/cat_rec_star/car_create_blog_v1_mode0.md")

        self.config_dict = {
            '6679129087': self.prompt_star,
            '6348148898': self.prompt_life,
            '5623715908': self.prompt_live,
            '5608272697': self.prompt_car
        }
        
    def get(self, key):
        return self.config_dict.get(key, self.prompt_default)
        
    def get_prompt(self, path):
        with open(path, encoding='utf-8') as f:
            base_prompt = f.read()
        return base_prompt
        
if __name__ == '__main__':
    model_config = Model_config()
    print(model_config.config_dict)
    sys.exit()

