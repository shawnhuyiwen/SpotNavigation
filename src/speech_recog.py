import speech_recognition as sr

class Recognizor:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
    
    def initiate_recognizor(self):
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source)
            audio = self.rec.listen(source)

        try:
            # print("Google Speech Recognition results:")
            return self.rec.recognize_google(audio, show_all=True, language="en-GB")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return "Google Speech Recognition could not understand audio"