import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data[:10]

def preprocess_function(data):
    question = data["qa"]["question"]
    answer = data["qa"]["answer"]
    # context = "".join(data["paragraphs"]).join(data["tables"]).join(data["table_description"])
    context = "".join(data["paragraphs"]).join(data["table_description"].values())
    return (str(question), str(answer), str(context))

def evaluate_gpt2_qa_model(model, tokenizer, data):
    correct_predictions = 0
    total_examples = len(data)

    for example in data:
        question, ground_truth_answer, context = preprocess_function(example)

        input_text = f"Question: {question} Context: {context}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation = True, max_length = 100)

        # Generate answer
        output = model.generate(input_ids, max_length=200, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

        # Decode and return the answer
        predicted_answer = tokenizer.decode(output[0], skip_special_tokens=True).replace("Answer:", "").strip()

        # Evaluate
        if predicted_answer.lower() == ground_truth_answer.lower():
            correct_predictions += 1
        print(predicted_answer.lower())
        print(ground_truth_answer.lower())
    accuracy = correct_predictions / total_examples
    return accuracy

# Load your BERT model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load your evaluation data from a JSON file
json_file_path = r"data\train.json"
evaluation_data = load_json(json_file_path)

# Evaluate the model
accuracy = evaluate_gpt2_qa_model(model, tokenizer, evaluation_data)

print(f"Accuracy: {accuracy * 100:.2f}%")
