import json
import os
from ragas import evaluate
from datasets import Dataset

from langchain.evaluation.qa import QAEvalChain
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv()) 
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

with open("/mnt/d/sravan/rag/CORD19/evaluation.json") as f:
    eval_data = json.load(f)

questions = []
contexts = []
answers = []
ground_truth = []

for data in eval_data['evaluation']:
    questions.append(data['Question'])
    contexts.append([data['Context']])
    answers.append(data['Ground_truth'])
    ground_truth.append(data['Answer'])
    
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
}

dataset = Dataset.from_dict(data)

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
)

df = result.to_pandas()

df.to_csv('/mnt/d/sravan/rag/CORD19/RAGAS_Evaluation_Result.csv', index=False)


examples = []
predictions = []
for data in eval_data['evaluation']:
    examples.append({'question' :data['Question'],
                     'answer' : data['Ground_truth']})
    predictions.append({'text':data['Answer']})

examples[0]
predictions[0]    


llm = OpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_output = eval_chain.evaluate(examples, predictions, question_key='question', answer_key = 'answer', prediction_key='text')

[{'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' INCORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}, {'results': ' INCORRECT'}, {'results': ' CORRECT'}, {'results': ' CORRECT'}]

no_of_correct_answers = 0
for results in graded_output:
    if results['results'] == ' CORRECT':
        no_of_correct_answers += 1
    
accuracy = no_of_correct_answers / len(graded_output)
0.928571
