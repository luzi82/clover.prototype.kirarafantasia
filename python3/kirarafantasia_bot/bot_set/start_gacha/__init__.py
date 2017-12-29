import kirarafantasia_bot

SCREEN_SIZE = 1280, 720
VIDEO_SIZE = kirarafantasia_bot.VIDEO_SIZE
TOUCH_SIZE = kirarafantasia_bot.TOUCH_SIZE

# import from kirarafantasia_bot.bot_set.start_gacha.bot_logic
BTN_SIZE = 60
def btn_rect(idx):
     return (SCREEN_SIZE[0]-(BTN_SIZE*(idx+1)),SCREEN_SIZE[1]-BTN_SIZE,SCREEN_SIZE[0]-(BTN_SIZE*idx),SCREEN_SIZE[1])
