from transformers import pipeline

filename = "experiment/europarl.experiment.fr"

def getFile():
    with open(filename, 'r', encoding='utf-8') as file:
        count = 1
        for line in file:
            print(count)
            yield "Translate French to English" + line
            count += 1
        file.close()

pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=256)

f = open("experiment/europarl.t5.result.en", "a")
for out in pipe(getFile()):
    f.write(out[0]["generated_text"] + "\n")
f.close()