from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Snowflake/Arctic-Text2SQL-R1-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = (
    "Given the table 'employees' with columns 'id', 'name', 'department', 'salary', "
    "write an SQL query to find the names of employees in the 'Engineering' department earning more than $100,000."
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(sql_query)
