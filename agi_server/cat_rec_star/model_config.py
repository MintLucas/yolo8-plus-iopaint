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
        self.prompt_new = self.get_prompt("config/prompt/cat_rec_star/new_create_blog_v1_mode0.md")
        self.prompt_exercise = self.get_prompt("config/prompt/cat_rec_star/exercise_create_blog_v1_mode0.md")

        
        self.config_dict = {
            'star': self.prompt_star,
            'life': self.prompt_life,
            'live': self.prompt_live,
            'car': self.prompt_car,
            'sport': self.prompt_sport,
            'new': self.prompt_new,
            'exercise': self.prompt_exercise
        }
        
    def get(self, key):
        trans_key = self.uid_map.get(key, "6679129087") if any(char.isdigit() for char in key) else key
        return self.config_dict.get(trans_key, self.prompt_default)
        
    def get_prompt(self, path):
        with open(path, encoding='utf-8') as f:
            base_prompt = f.read()
        return base_prompt
        
if __name__ == '__main__':
    PK_PROMPT_TEMPLATES = {
    # --- 你之前的风格优化版 ---
    "arcade_metallic": (
        "街机格斗游戏风格的 UI 界面，画面中央是两个巨大的圆形头像框，"
        "左边的头像框里是 {role_a}，右边的头像框里是 {role_b}，身体可部分超出框外。"
        "头像框有金属质感，边缘有发光特效。背景是深色渐变带像素粒子。"
        "中间有醒目的 'VS' 标识，整体充满竞技感。"
    ),
    "pixel_16bit_classic": (
        "16-bit 像素风街机 PK 界面，画面中央是分割的擂台。左侧角色是 {role_a}，"
        "右侧角色是 {role_b}，角色脚下有代表能力的属性数值。背景左侧和右侧对应各自的主题元素，"
        "下方有复古进度条 UI，高饱和度，带红白机粒子特效。"
    ),
    "tang_pixel_style": (
        "16-bit 像素风唐风街机 PK 界面，暮色紫红渐变天际，中央是青砖八角擂台。"
        "左侧角色是 {role_a}（穿着古风印花服饰），右侧角色是 {role_b}（携带古风道具）。"
        "上方悬浮卷轴计分板，整体色调饱和，带像素粒子特效。"
    ),
    "cartoon_3d_vibrant": (
        "卡通3D立体风格，造型圆润，颜色鲜亮。街机格斗 UI 界面。"
        "左侧是角色 {role_a}，右侧是角色 {role_b}。中间有巨大的 3D 质感 'VS' 标识，"
        "背景是梦幻的竞技场，光影明快。"
    ),
    "american_cartoon_ui": (
        "美式卡通风格（线条粗黑圆润，高饱和色块）。街机格斗游戏 UI 界面，"
        "画面中央是两个巨大的方形头像框，左边是 {role_a}，右边是 {role_b}。"
        "背景是夸张的漫画爆炸效果，中间有手绘风格的 'VS' 字样。"
    ),

    # --- 新增：基于游戏经历的醒目风格 ---
    
    "cyberpunk_glitch": (
        "赛博朋克科技风格。画面被霓虹灯管一分为二，左侧是青蓝色调的 {role_a}，"
        "右侧是玫红色调的 {role_b}。背景是充满故障艺术（Glitch Art）的电子屏幕，"
        "带有扫描线效果。中间是发光的电子 'VS' 标志，极具未来感。"
    ),
    "tcg_card_battle": (
        "集换式卡牌对战风格（类似炉石或王牌对决）。画面中有两张精美的立体卡牌浮现，"
        "左边卡牌的主角是 {role_a}，右边卡牌的主角是 {role_b}。卡牌边缘有金色的稀有度特效。"
        "背景是古老的木质桌面，散落着一些金币，充满策略对决感。"
    ),
    "anime_split_screen": (
        "热血日漫斜切分屏风格。画面被一条锐利的斜线切开，左上方是 {role_a} 的特写，"
        "右下方是 {role_b} 的特写。背景布满了动感的黑白速度线，角色周围有燃起的火焰斗气。"
        "中间横跨一个巨大的红色 'VS'，视觉冲击力极强。"
    ),
    "ink_martial_arts": (
        "传统水墨写意风格。画面中央是一道苍劲的墨痕。左侧是水墨晕染出的 {role_a}，"
        "右侧是笔触凌厉的 {role_b}。背景是留白的宣纸质感，点缀红色印章装饰。"
        "中间用书法字体写着 '对决'（或 VS），风格硬朗且具有东方韵味。"
    ),
    "boss_health_bar": (
        "横版 RPG 终极 Boss 战风格。画面正中央是 {role_a} 与 {role_b} 的对峙状态。"
        "屏幕上方和下方分别显示两人的超长血条和名字。背景是史诗感的荒野战场，"
        "天空中雷电交加，营造出一种‘最后一战’的压迫感。"
    )
}
    tt = {"role_a": "tt1","role_b": "tt2"}
    PK_PROMPT_TEMPLATES['boss_health_bar'].format(**tt)
    model_config = Model_config()
    print(model_config.config_dict)
    model_config.init_all_prompt()
    print(model_config.get("5608272697"))
    sys.exit()

