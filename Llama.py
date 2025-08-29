import torch
from transformers import pipeline
from huggingface_hub import login
import time
aToken = ""
log = login(
    token=aToken
)

filename = "experiment/europarl.experiment.fr"

initMessage =  {"role": "system", "content": "You are a French to English Translator, translate the input sentences and only give the output sentence"}

def getFile():
    #inputs = []
    with open(filename, 'r', encoding='utf-8') as file:
        count = 1
        for line in file:
            print(count)
            yield [initMessage, {"role": "user", "content": line}]
            count += 1
        file.close()

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    return_full_text=False,
    max_new_tokens=300
)

start = time.time()

file = open("experiment/europarl.llama.result1.en", "w")

end = time.time()

for out in pipe(getFile()):
    file.write(out[0]['generated_text'] + "\n")

print("total time: ", end-start)