import sys,os
sys.path.append(os.getcwd())
from util.mylogging import get_logger


class Model_config:
    def __init__(self, log=get_logger("model_config")):
        self.log = log
        self.prompt_default = self.get_prompt("config/prompt/cat_rec_star/create_blog_v4_mode0.md")
        self.config_dict = {}
        self.init_all_prompt()
        self.uid_map = {
            '6679129087': 'star',
            '6348148898': 'life',
            '5623715908': 'live',
            '5608272697': 'car',
            '6320391439': 'sport',
        }
    def init_all_prompt(self):

        self.prompt_star = self.get_prompt("config/prompt/cat_rec_star/create_blog_v4_mode0.md")
        self.prompt_life = self.get_prompt("config/prompt/cat_rec_star/life_create_blog_v1_mode0.md")
        self.prompt_live = self.get_prompt("config/prompt/cat_rec_star/live_create_blog_v1_mode0.md")
        self.prompt_car = self.get_prompt("config/prompt/cat_rec_star/car_create_blog_v1_mode0.md")
        self.prompt_sport = self.get_prompt("config/prompt/cat_rec_star/sport_create_blog_v1_mode0.md")

        
        self.config_dict = {
            'star': self.prompt_star,
            'life': self.prompt_life,
            'live': self.prompt_live,
            'car': self.prompt_car,
            'sport': self.prompt_sport,
        }
        
    def get(self, key):
        trans_key = self.uid_map.get(key, "6679129087") if any(char.isdigit() for char in key) else key
        return self.config_dict.get(trans_key, self.prompt_default)
        
    def get_prompt(self, path):
        with open(path, encoding='utf-8') as f:
            base_prompt = f.read()
        return base_prompt
        
if __name__ == '__main__':
    model_config = Model_config()
    print(model_config.config_dict)
    model_config.init_all_prompt()
    print(model_config.get("5608272697"))
    sys.exit()

