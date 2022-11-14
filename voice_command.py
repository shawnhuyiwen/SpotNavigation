from speech_recog import Recognizor
import nltk
from nltk.stem.snowball import SnowballStemmer

commands = {
    'sit': 0,
    'forward': 1,
    'backwards': 2,
    'left': 3,
    'right': 4
}

rec = Recognizor()

results = rec.initiate_recognizor()

def preprocessing(results:list):
    result_list = []
    for result in results['alternative']:
        result_list.append(result['transcript'])

    tokenizer = nltk.RegexpTokenizer(r"\w+") 

    tok_results = []
    for result in result_list:
        tok_results.append(tokenizer.tokenize(result.lower()))
    
    print(tok_results)

    # sb_stemmer = SnowballStemmer('english')
    # stemmed_results = []
    # for result in tok_results:
    #     stemmed_results.append([sb_stemmer.stem(word) for word in result])

    return tok_results

def check_commands(commands, keywords):
    for keyword in keywords:
        if keyword in commands[0]:
            return True, keyword
        for command in commands:
            if keyword in command:
                return True, keyword
    
    return False, ''



# print(preprocessing(results))
