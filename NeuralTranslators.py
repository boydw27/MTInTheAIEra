from transformers import pipeline

filename = "experiment/europarl.experiment.fr"

class MarianTranslator:

    def getFile(self):
        with open(filename, 'r', encoding='utf-8') as file:
            count = 1
            for line in file:
                print(count)
                yield line
                count += 1
            file.close()

    def __init__(self):
        self.pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
        
    def translate(self):
        f = open("experiment/europarl.marian.results.en", "a")
        for out in self.pipe(self.getFile()):
            f.write(out[0]["translation_text"] + "\n")
        f.close()


class M2M100:

    def getFile(self):
        with open(filename, 'r', encoding='utf-8') as file:
            count = 1
            for line in file:
                print(count)
                yield line
                count += 1
            file.close()

    def __init__(self):
        self.pipe = pipeline("translation", "facebook/m2m100_1.2B", src_lang="fr", tgt_lang="en")
    
    def translate(self):
        f = open("experiment/europarl.m2m100.results.en", "a")
        for out in self.pipe(self.getFile()):
            f.write(out[0]["translation_text"] + "\n")
        f.close()
    
class NLLB:

    def getFile(self):
        with open(filename, 'r', encoding='utf-8') as file:
            count = 1
            for line in file:
                print(count)
                yield line
                count += 1
            file.close()

    def __init__(self):
        self.pipe = pipeline("translation", model="facebook/nllb-200-1.3B", src_lang="fra_Latn", tgt_lang="eng_Latn")
    
    def translate(self):
        f = open("experiment/europarl.nllb.results.en", "a")
        for out in self.pipe(self.getFile()):
            f.write(out[0]["translation_text"] + "\n")
        f.close()


if __name__ == "__main__":
    marian = MarianTranslator()
    marian.translate()
    del marian
    m2m = M2M100()
    m2m.translate()
    del m2m
    nllb = NLLB()
    nllb.translate()
    del nllb