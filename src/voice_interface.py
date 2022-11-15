from speech_recog import Recognizor
from text_processing import TextProcessor
import time
import warnings

warnings.filterwarnings('ignore')

is_activate = False
commands = {
    'sit': 0,
    'down': 0,
    'forward': 1,
    'backwards': 2,
    'left': 3,
    'right': 4
}

while True:
    print("I'm listening")
    rec = Recognizor()
    text_precessor = TextProcessor()
    results = rec.initiate_recognizor()

    # print(results)

    if(type(results) == str or results == []):
        continue

    text_precessor.preprocessing(results)
    
    if not is_activate:
        is_activate = text_precessor.check_commands(['spot', 'sport'])[0]
        time.sleep(0.5)
        if is_activate == True:
            print("I'm here!")
    else:
        # detected, command = check_commands(pr_text, ['sit', 'down','forward','backwards','left','right'])
        # if detected:
        #     print(commands[command])
        
        dest = text_precessor.extract_dest()
        print(dest)
        


    
