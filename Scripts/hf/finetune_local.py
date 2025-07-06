from transformers import AutoModelForCausalLM, AutoTokenizer
import os

root_dir = os.getenv("SN3_ROOT", default=os.getcwd())


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model_name = f"{root_dir}/Models/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def make_prompt(line):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{"Answer the following question about WordNet synsets." if line['type'] == 'QA' else "Finish statement about WordNet synsets."}
<|start_header_id|>user<|end_header_id|>
{line['in']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{line['out']}<|eot_id|>"""

from datasets import load_dataset

# ds = load_dataset("VityaVitalich/WordNet-TaxoLLaMA")
ds = load_dataset("json", data_files={"train": f"{root_dir}/Data/wn-3.0-json/dataset.jsonl"})

# sample = ds["train"][0]
# prompt = make_prompt(sample)
# tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

# for k, v in tokens.items():
#     print(f"{k}: shape {v.shape}")

# exit()

tokenized = ds.map(
    lambda examples: tokenizer(make_prompt(examples), truncation=True, max_length=512),
    batched=True,
    remove_columns=["in", "out", "synsets", "variant", "type"],
)


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

kbit_model = prepare_model_for_kbit_training(base_model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Depending on architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(kbit_model, config)

print("base_model = ", base_model)
print("kbit_model = ", kbit_model)
print("peft_model = ", model)
# print("tokenizer = ", tokenizer)


from transformers import Trainer, TrainingArguments
import datetime as dt

training_args = TrainingArguments(
    output_dir=f"{root_dir}/Models/Llama-3.1-8B-Instruct-{dt.datetime.now().strftime('%y%m%d-%H%M-%S')}",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"]
)

trainer.train()