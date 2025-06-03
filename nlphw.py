from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


tokenizer = AutoTokenizer.from_pretrained('gpt2')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt1 = 'It was a bright and sunny'
prompt2  ='She opened the book and'

ids1 = tokenizer(prompt1, return_tensors='pt').input_ids
for id in ids1:
  print(id, '\t:',tokenizer.decode(id))

ids2 = tokenizer(prompt2, return_tensors='pt').input_ids
for id in ids2:
  print(id, '\t:', tokenizer.decode(id))



gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
outputs1 = gpt2(ids1)
outputs2 = gpt2(ids2)

greedy_output1 = gpt2.generate(ids1, max_new_tokens=20)
print(tokenizer.decode(greedy_output1[0]))

greedy_output2 = gpt2.generate(ids2, max_new_tokens=20)
print(tokenizer.decode(greedy_output2[0]))


output_ids1 = gpt2.generate(ids1, max_new_tokens = 20, do_sample=True, top_k=100, top_p=0.9, temperature=1)
print(output_ids1)
tokenizer.decode(output_ids1[0])

output_ids2 = gpt2.generate(ids2, max_new_tokens = 20, do_sample=True, top_k=100, top_p=0.9, temperature=1)
print(output_ids2)
tokenizer.decode(output_ids2[0])


