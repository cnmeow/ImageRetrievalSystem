import translate
import googletrans


class Translator():
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'google':
            self.model = googletrans.Translator()
        elif mode in 'translate':
            self.model = translate

    def preprocessing(self, text):
        return text.lower()

    def __call__(self, text):
        text = self.preprocessing(text)
        return self.model.translate(text) if self.__mode in 'translate' else self.model.translate(text, src=self.__from_lang, dest=self.__to_lang).text