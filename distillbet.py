import json
from pathlib import Path
import nltk
import urllib.request
import urllib
#!nltk.download('reuters')

"""Get the Transformers Library, this has the Tokenizer and QA pre trained model for DistilBert"""

#!pip install transformers

def read_ipc(data_dict, split):
    '''
    this method will take in the data annotated by CDQA
    and split the data, create a test and train split.
    '''
    # squad_dict=data_dict
    contexts = []
    questions = []
    answers = []
    train_contexts = []
    train_questions = []
    train_answers = []
    val_contexts = []
    val_questions = []
    val_answers = []

    #### for these following contexts(paras), I typed out the answers
    #### instead of selecting a span in context, Hence the annotator had to
    #### select start_token as -1. We will skip them.

    wrong_answers=[15,29,35,39]

    for x,group in enumerate(data_dict['data']):
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    if answer['answer_start']!=-1:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

    ##### splitting wrt number split given#####
    train_end=int(len(contexts)*split)
    ## 0 to 294

    for x in range(train_end):
        train_contexts.append(contexts[x])
        train_questions.append(questions[x])
        train_answers.append(answers[x])
    print(len(train_answers))
    ### 295 to end of doc
    for x in range(train_end,len(contexts)):
        val_contexts.append(contexts[x])
        val_questions.append(questions[x])
        val_answers.append(answers[x])

    print(len(val_answers))
    return train_contexts, train_questions, train_answers,val_contexts, val_questions, val_answers

f = open('cdqa_ipc.json')
data = json.load(f)

"""Let's take a look at the first IPC Chapters, our dataset"""

print("\n".join([x["context"] for x in data["data"][0]["paragraphs"]]))

"""Great, now let's see how the dataset is structured, in terms of labelling of questions and answers"""

data["data"][0]

print("QA pair on first chapter of IPC")
for qa_pair in data["data"][0]["paragraphs"]:
  print("context (para/heading in IPC chapter)"+str(qa_pair["context"]))
  print("Question on context: "+str(qa_pair["qas"][0]["question"]))
  print("answer from context: "+str(qa_pair["qas"][0]["answers"][0]["text"]))
  print("\n")

train_contexts, train_questions, train_answers,val_contexts, val_questions, val_answers = read_ipc(data,0.8)

print((train_contexts[0]))
print((train_questions[0]))
print((train_answers[0]))

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

print(tokenizer)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    print(len(answers))
    for i in range(len(answers)):
        try:
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        except:
            print(answers[i],i)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

import torch

class IpcDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = IpcDataset(train_encodings)
val_dataset = IpcDataset(val_encodings)

############# CONSIDER FROM HERE

from transformers import DistilBertForQuestionAnswering
model_db = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

'''this cell is code for fine tuning'''

from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_db.to(device)
model_db.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model_db.parameters(), lr=5e-5)

for epoch in range(10):
    for x,batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model_db(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
        if x == 10:
            print("batch: "+str(x)+" loss: "+str(loss))
model_db.eval()

val_questions[1]

# question="what is the punishment for preparing to commit dacoity?"
question = val_questions[1]
# paragraph="Whoever makes any preparation for committing dacoity, shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine."
# val_contexts[2]
paragraph = val_contexts[2]
ans=val_answers[2]
# val_answers[2]
print(question)
print(paragraph)
print(ans)

encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)
input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
tokens = tokenizer.convert_ids_to_tokens(input_ids) #input tokens
print(tokens)
# print(input_ids, attention_mask)

print(tokenizer)

ip=torch.tensor([input_ids]).to(device)
attention=torch.tensor([attention_mask]).to(device)

output = model_db(ip, attention)
start_scores = output.start_logits
end_scores = output.end_logits

max_startscore = torch.argmax(start_scores)
max_endscore = torch.argmax(end_scores)
ans_tokens = input_ids[max_startscore: max_endscore + 1]
# answer = ' '.join(tokens[start_index:end_index+1])
print(ans_tokens)
answer = ' '.join(tokens[max_startscore:max_endscore+1])
print("question: "+str(question))
print("answer: "+str(answer))
print("context: "+str(paragraph))