!pip install transformers[torch]
import torch
import transformers
from transformers import AutoTokenizer, OpenAIGPTLMHeadModel

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
model = transformers.OpenAIGPTLMHeadModel(transformers.OpenAIGPTConfig(n_positions=3000, vocab_size=40479, n_layers=1))

import pandas as pd
path=""
data=pd.read_csv("DATAPATH")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.model_max_length=3000

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=10)

import torch
import glob
class CDataset(torch.utils.data.Dataset):
    def __init__(self, data):
      self.data=data



    def __getitem__(self, idx):
        t={'input_ids': torch.tensor(tokenizer(self.data["1"][idx], padding='max_length', truncation=True).input_ids), 'attention_mask': torch.tensor(tokenizer(self.data["1"][idx], padding='max_length', truncation=True).attention_mask), 'labels': torch.tensor(tokenizer(self.data["0"][idx], padding='max_length', truncation=True).input_ids)}
        #item=dict()
        #item['labels'] = torch.tensor(tokenizer(self.data["0"][idx], return_tensors="pt", padding="max_length").input_ids).clone().detach()
        #item["input"] = torch.tensor(tokenizer(self.data["1"][idx], return_tensors="pt", padding="max_length").input_ids).clone().detach()
        return t

    def __len__(self):
        return len(self.data)

train_dataset = CDataset(data)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="/kaggle/working/",          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    weight_decay=0.01,               # strength of weight decay
    logging_dir="/kaggle/working/",
    gradient_accumulation_steps=1,# directory for storing logs
    logging_steps=1,
    prediction_loss_only=True
)


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
)

trainer.train()
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

prompt = (
    "Lets do our work"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(input_ids)[0]
print(gen_text)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.max_length=100000

prompt = (
    "Lets do our work by this way"
)

input_ids = tokenizer(prompt, return_tensors="pt")
print(input_ids)
