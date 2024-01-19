import json
from transformers import BertTokenizer, BertForQuestionAnswering
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

def evaluate_bert_qa_model(model, tokenizer, data):
    correct_predictions = 0
    total_examples = len(data)

    for example in data:
        question, ground_truth_answer, context = preprocess_function(example)

        inputs = tokenizer(question, context, return_tensors='pt', truncation = True, max_length = 512)
        
        start_positions = tokenizer(ground_truth_answer, return_tensors='pt')['input_ids']

        # Forward pass
        outputs = model(**inputs, start_positions=start_positions)
        start_logits = outputs.start_logits

        # Get the most probable answer
        start_index = start_logits.argmax(dim=1).item()
        predicted_answer = tokenizer.decode(inputs['input_ids'][0, start_index:].tolist())

        # Evaluate
        if predicted_answer.lower() == ground_truth_answer.lower():
            correct_predictions += 1
        print(predicted_answer.lower())
        print(ground_truth_answer.lower())
    accuracy = correct_predictions / total_examples
    return accuracy

# Load your BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load your evaluation data from a JSON file
json_file_path = r"data\train.json"
evaluation_data = load_json(json_file_path)

# Evaluate the model
accuracy = evaluate_bert_qa_model(model, tokenizer, evaluation_data)

print(f"Accuracy: {accuracy * 100:.2f}%")
