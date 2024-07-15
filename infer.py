from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import json

file = open("./edited-only-sc/edited-sc-12.json", "r")
content = json.load(file)

print(content["scorecard"].__str__())


tokenizer = T5Tokenizer.from_pretrained("./trained-flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("./trained-flan-t5-base")

def generate_summary(input_text):
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation= True)
    outputs = model.generate(inputs, max_length=150, min_length=30, Length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


file_path = os.path.join("C:/Users/studd/Downloads/articles/articles", "article1.txt")
file = open(file_path, "r")

file_content = json.load(file)
input_text=file_content['match_info']
print(file_content)
summary = generate_summary(input_text)
print(summaгу)

#input_ids = tokenizer(content["scorecard"].__str__(), return_tensors="pt").input_ids
#outputs = model.generate(input_ids)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))