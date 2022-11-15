import nltk
import spacy


class TextProcessor:
    
    def __init__(self):
        self.nlp_model = spacy.load("./models/model-best")
        self.text_list = []
        self.tok_list = []
    
    def preprocessing(self, results):
        for result in results['alternative']:
            self.text_list.append(result['transcript'])

        tokenizer = nltk.RegexpTokenizer(r"\w+") 

        for result in self.text_list:
            self.tok_list.append(tokenizer.tokenize(result.lower()))
        
        # print(self.tok_list)

        return self.tok_list

    def check_commands(self, keywords):
        for keyword in keywords:
            if keyword in self.tok_list[0]:
                return True, keyword
            for command in self.tok_list:
                if keyword in command:
                    return True, keyword
        
        return False, ''
    
    def extract_dest(self):
        for text in self.text_list:
            doc = self.nlp_model(text)
            for ent in doc.ents:
                if ent.label_ == "DEST":
                    return str(ent.text)
                
        return "undetected"
        
        
