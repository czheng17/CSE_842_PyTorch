### Chen Zheng 05/19/2022

import torch
from torch import nn
import time
import torchtext
import numpy as np
import numpy

import random

from collections import defaultdict, Counter


from data_processing import read_data, evaluate_iob, prf
from model import RNNTagger

import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina' 
plt.style.use('seaborn')

class Tagger:
    
    def __init__(self, lower):
        self.TEXT = torchtext.legacy.data.Field(init_token='<bos>', eos_token='<eos>', sequential=True, lower=lower)
        self.LABEL = torchtext.legacy.data.Field(init_token='<bos>', eos_token='<eos>', sequential=True, unk_token=None)
        self.fields = [('text', self.TEXT), ('label', self.LABEL)]
        # self.device = "cuda:0"
        self.device = "cpu"
        
    def tag(self, sentence):
        labels = ['O']*len(sentence) # placeholder
        example = torchtext.legacy.data.Example.fromlist([sentence, labels], self.fields)
        dataset = torchtext.legacy.data.Dataset([example], self.fields)
        
        iterator = torchtext.legacy.data.Iterator(
            dataset,
            device=self.device,
            batch_size=1,
            repeat=False,
            train=False,
            sort=False)
        
        # Apply the trained model to all batches.
        out = []
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                predicted = self.model.predict(batch.text)[0][0]

                label_list = ['<pad>', '<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']
                # label_list = ['<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']
                out = []
                for pred in predicted:
                    out.append(label_list[pred])

        return out[1:]


    def train(self):
        # Read training and validation data according to the predefined split.
        train_examples = read_data('data/eng.train', self.fields)
        valid_examples = read_data('data/eng.testa', self.fields)
        test_examples = read_data('data/eng.testb', self.fields)

        # Count the number of words and sentences.
        n_tokens_train = 0
        n_sentences_train = 0
        for ex in train_examples:
            n_tokens_train += len(ex.text) + 2
            n_sentences_train += 1
        n_tokens_valid = 0       
        for ex in valid_examples:
            n_tokens_valid += len(ex.text)
        
        n_tokens_test = 0       
        for ex in test_examples:
            n_tokens_test += len(ex.text)

        # Load the pre-trained embeddings that come with the torchtext library.
        use_pretrained = True
        if use_pretrained:
            print('We are using pre-trained word embeddings.')
            self.TEXT.build_vocab(train_examples, vectors="glove.840B.300d")
        else:  
            print('We are training word embeddings from scratch.')
            self.TEXT.build_vocab(train_examples, max_size=5000)
        self.LABEL.build_vocab(train_examples)
    
        # Create one of the models defined above.
        self.model = RNNTagger(self.TEXT, self.LABEL, emb_dim=300, rnn_size=128, update_pretrained=False)
        # self.model = RNNCRFTagger(self.TEXT, self.LABEL, emb_dim=300, rnn_size=128, update_pretrained=False)
    
        self.model.to(self.device)
    
        batch_size = 1024
        n_batches = np.ceil(n_sentences_train / batch_size)

        mean_n_tokens = n_tokens_train / n_batches

        train_iterator = torchtext.legacy.data.BucketIterator(
            train_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=True,
            sort=True)

        valid_iterator = torchtext.legacy.data.BucketIterator(
            valid_examples,
            device=self.device,
            batch_size=64, ### 64
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=False,
            sort=True)

        test_iterator = torchtext.legacy.data.BucketIterator(
            valid_examples,
            device=self.device,
            batch_size=1,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=False,
            sort=True)   
        
    
        train_batches = list(train_iterator)
        valid_batches = list(valid_iterator)
        test_batches = list(test_iterator)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-5)

        n_labels = len(self.LABEL.vocab)

        history = defaultdict(list)    
        
        n_epochs = 20
        
        for i in range(1, n_epochs + 1):

            t0 = time.time()

            loss_sum = 0

            self.model.train()
            for batch in train_batches:
                
                # Compute the output and loss.
                loss = self.model(batch.text, batch.label) / mean_n_tokens
                
                optimizer.zero_grad()            
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            train_loss = loss_sum / n_batches
            history['train_loss'].append(train_loss)

            # Evaluate on the validation set.
            if i % 1 == 0:
                stats = defaultdict(Counter)

                self.model.eval()
                with torch.no_grad():
                    for batch in valid_batches:
                        # Predict the model's output on a batch.
                        predicted, tmp_scores = self.model.predict(batch.text)                   
                        # Update the evaluation statistics.
                        evaluate_iob(predicted, batch.label, self.LABEL, stats)
            
                # Compute the overall F-score for the validation set.
                _, _, val_f1 = prf(stats['total'])
                
                history['val_f1'].append(val_f1)
            
                t1 = time.time()
                print(f'Epoch {i}: train loss = {train_loss:.4f}, val f1: {val_f1:.4f}, time = {t1-t0:.4f}')
           

        self.model.eval()
        with torch.no_grad():
            for batch in test_batches:
                # Predict the model's output on a batch.
                predicted, tmp_scores = self.model.predict(batch.text)
                # print(predicted)                  
                # Update the evaluation statistics.
                evaluate_iob(predicted, batch.label, self.LABEL, stats)

        print('Final evaluation on the test set:')
        p, r, f1 = prf(stats['total'])
        print(f'Overall: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        for label in stats:
            if label != 'total':
                p, r, f1 = prf(stats[label])
                print(f'{label:4s}: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')

        stats = defaultdict(Counter)
        label_list = ['<pad>', '<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']
        # label_list = ['<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']

        self.model.eval()
        with torch.no_grad():
            for batch in test_batches:
                # Predict the model's output on a batch.
                predicted_tagging, predicted_score = self.model.predict(batch.text)
                evaluate_iob(predicted_tagging, batch.label, self.LABEL, stats)

        print('Final evaluation on the test set:')
        p, r, f1 = prf(stats['total'])
        print(f'Overall: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        for label in stats:
            if label != 'total':
                p, r, f1 = prf(stats[label])
                print(f'{label:4s}: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')

tagger = Tagger(lower=False)
tagger.train()


def print_tags(sentence):
    tokens = sentence.split()
    tags = tagger.tag(tokens)
    for token, tag in zip(tokens, tags):
        print(f'{token:12s}{tag}')


print('The first case: John Johnson was born in Moscow , lives in Gothenburg , and works for Chalmers Technical University and the University of Gothenburg .')
print_tags('John Johnson was born in Moscow , lives in Gothenburg , and works for Chalmers Technical University and the University of Gothenburg .')

print('The second case: Paris Hilton lives in New York .')
print_tags('Paris Hilton lives in New York .')

print('The third case: Paris Hilton lives in New York .')
print_tags('New York Stock Exchange is in New York .')