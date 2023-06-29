# importing libraries

from flask import Flask, jsonify, request
import requests
import json
from dotenv.main import load_dotenv
import transformers
import os
import re
import pandas as pd
import torch
import csv
import numpy as np
from transformers import pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

# model name

checkpoint = "roberta-base"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

# Creating the  Dataset Class

label_all_tokens = False


def align_label(texts, labels):
    text_arr = texts.split(" ")
    tokenized_inputs = tokenizer(text_arr, padding='max_length', max_length=512, truncation=True,
                                 is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        for ind, txt_upper in enumerate(txt):
            txt[ind] = txt_upper.lower()
        self.texts = [tokenizer(str(i),
                                padding='max_length', max_length=512, truncation=True, return_tensors="pt") for i in
                      txt]
        self.labels = [align_label(i, j) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


# labelling

labels_to_ids = {'I-NAME': 0, 'I-VALUE': 1, 'I-INTENT': 2, 'O': 3, 'B-NAME': 4, 'B-VALUE': 5, 'B-INTENT': 6}
ids_to_labels = {0: 'I-NAME', 1: 'I-VALUE', 2: 'I-INTENT', 3: 'O', 4: 'B-NAME', 5: 'B-VALUE', 6: 'B-INTENT'}


# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),[int(.7 * len(df)), int(0.9 * len(df))])


# Build Model
class BertModel(torch.nn.Module):

    def __init__(self):
        super(BertModel, self).__init__()

        self.bert = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


'''
# plots
x = []
trl = []
vl = []
tra = []
vla = []
'''


# Model training
def train_loop(model, df_train, df_val):
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=2, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)
        x.append(epoch_num + 1)
        trl.append(total_loss_train / len(df_train))
        vl.append(total_loss_val / len(df_val))
        tra.append(total_acc_train / len(df_train))
        vla.append(total_acc_val / len(df_val))

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')


'''
# parameters
LEARNING_RATE = 0.005
EPOCHS = 5
BATCH_SIZE = 2
'''


# model = BertModel()
# train_loop(model, df_train, df_val)


# Evaluate Model
def evaluate(models, df_test):
    test_dataset = DataSequence(df_test)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        models = models.cuda()

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:

        test_label = test_label.to(device)
        mask = test_data['attention_mask'].squeeze(1).to(device)

        input_id = test_data['input_ids'].squeeze(1).to(device)

        loss, logits = models(input_id, mask, test_label)

        for i in range(logits.shape[0]):
            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')


# Predict One Sentence
def align_word_ids(texts):
    text_arr = texts.split(" ")
    tokenized_inputs = tokenizer(text_arr, padding='max_length', max_length=512, truncation=True,
                                 is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


# function to get the most probabilistic entities
def fn(input_txt, iob_tag):
    print(iob_tag)
    tokens = input_txt.split(" ")
    objtmp = {
        'INTENT': [],
        'NAME': [],
        'VALUE': []
    }
    objtmpsecond = {
        'INTENT': [],
        'NAME': [],
        'VALUE': []
    }
    objf = {
        'text': input_txt,
        'INTENTroberta': '',
        'NAMEroberta': '',
        'VALUEroberta': '',
    }
    length_iob = len(iob_tag)
    curr = 0
    while curr < length_iob:
        print(curr,objtmp)
        if iob_tag[curr][0] == 'O':
            curr += 1
            continue
        else:
            tag = iob_tag[curr][0][2:]
            if iob_tag[curr][0][0] == 'B':
                temp_arr = [str(iob_tag[curr][1]), tokens[curr]]
                chk = 0
                for ind in range(curr + 1, length_iob):
                    if iob_tag[ind][0] == 'O' or iob_tag[ind][0][0] == 'B' or iob_tag[ind][0][2:] != tag:
                        objtmp[tag].append(temp_arr)
                        curr = ind
                        chk = 1
                        break
                    else:
                        chk = 2
                        curr = ind
                        temp_arr.append(tokens[ind])
                if chk == 0 or chk == 2:
                    curr += 1
                    objtmp[tag].append(temp_arr)
            else:
                curr += 1

    print(objtmp)
    for key in objtmp:
        lenkey = len(objtmp[key])
        if lenkey == 0:
            continue
        objtmp[key].sort()
        objtmpsecond[key] = objtmp[key][lenkey - 1][1:]
    print(objtmpsecond)
    objf['INTENTroberta'] = " ".join(objtmpsecond['INTENT'])
    objf['NAMEroberta'] = " ".join(objtmpsecond['NAME'])
    objf['VALUEroberta'] = " ".join(objtmpsecond['VALUE'])
    return objf
    # df.at[index,'labels'] = targetColumn


def evaluate_one_text(model, sentence):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    sentence = sentence.strip()
    sentence = re.sub(' +', ' ', sentence)
    sentence_lower = sentence.lower()
    text = tokenizer(sentence_lower.split(" "), padding='max_length', max_length=512, truncation=True,
                     return_tensors="pt", is_split_into_words=True)
    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence_lower)).unsqueeze(0).to(device)
    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]
    input_ids = input_id[label_ids != 100]
    predictions = logits_clean.argmax(dim=1).tolist()
    probabilities = torch.nn.functional.softmax(logits_clean, dim=1)
    prediction_label = [[ids_to_labels[index], round(probabilities[ite][index].item() * 100, 2)] for ite, index in
                        enumerate(predictions)]
    return fn(sentence, prediction_label)


# loading the fine-tuned model

model = torch.load('roberta-automator-modified.pt', map_location=torch.device('cpu'))

op = pd.read_csv("stepsOnly.csv",encoding='utf-8')
mydict = []
op = op.reindex(columns = op.columns.tolist() + ['INTENTroberta'] + ['NAMEroberta'] + ['VALUEroberta'])

for index, row in op.iterrows():
    input = row["text"]
    obj = evaluate_one_text(model,input)
    mydict.append(obj)

# field names
fields = ['text', 'INTENTroberta', 'NAMEroberta', 'VALUEroberta', ]

# name of csv file
filename = "Output.csv"

# writing to csv file
with open(filename, 'w', encoding="utf-8") as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames=fields)

    # writing headers (field names)
    writer.writeheader()

    # writing data rows
    writer.writerows(mydict)

app = Flask(__name__)
text_data = [
]


# home route
@app.route('/')
def home():
    return "bert ner"


# post a text and respective ner returned
@app.route('/bert/new', methods=['POST'])
def create_ner():
    request_data = request.get_json()
    new_data = {
        'input': request_data['input'],
        'output': evaluate_one_text(model, request_data['input'])
    }

    text_data.append(new_data)
    return jsonify(new_data)


# get all text results
@app.route('/bert')
def get_ner():
    return jsonify({'text_data': text_data})


app.run(port=8000)
