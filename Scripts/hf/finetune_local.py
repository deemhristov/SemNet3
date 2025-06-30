from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

tokenizer.save_pretrained("hf/models/llama3.1-8b-instruct")
model.save_pretrained("hf/models/llama3.1-8b-instruct") 