### Chen Zheng 05/19/2022

# Implementing a tagger based on RNNs and a linear output unit
# Our first implementation will be fairly straightforward. We apply an RNN and then a linear output unit to predict the outputs. 
# The following figure illustrates the approach. (The figure is a bit misleading here, because we are predicting BIO labels and not part-of-speech tags, 
# but you get the idea.)

# High-quality systems that for tasks such as named entity recognition and part-of-speech tagging typically use smarter word representations, 
# for instance by taking the characters into account more carefully. We just use word embeddings.

# A small issue to note here is that we don't want the system to spend effort learning to tag the padding tokens. 
# To make the system ignore the padding, we add a large number to the output corresponding to the dummy padding tag. 
# This means that the loss values for these positions will be negligible.

# Note that we structure the code a bit differently compared to our previous implementations: 
# we compute the loss in the forward method, while previously we just computed the output in this method. 
# The reason for this change is that the CRF (see below) uses this structure, and we want to keep the implementations compatible. 
# Similarly, the predict method will convert from PyTorch tensors into NumPy arrays, in order to be compatible with the CRF's prediction method.

import torch
from torch import nn
import numpy as np

class RNNTagger(nn.Module):
    
    def __init__(self, text_field, label_field, emb_dim, rnn_size, update_pretrained=False):
        super().__init__()
        
        voc_size = len(text_field.vocab)
        self.n_labels = len(label_field.vocab)       
        
        # Embedding layer. If we're using pre-trained embeddings, copy them
        # into our embedding module.
        self.embedding = nn.Embedding(voc_size, emb_dim)
        if text_field.vocab.vectors is not None:
            self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
                                                       requires_grad=update_pretrained)

        # RNN layer. We're using a bidirectional GRU with one layer.
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, 
                          bidirectional=True, num_layers=1)

        # the RNN size since we are using a bidirectional RNN.
        self.top_layer = nn.Linear(2*rnn_size, self.n_labels)
 
        # To deal with the padding positions later, we need to know the
        # encoding of the padding dummy word and the corresponding dummy output tag.
        self.pad_word_id = text_field.vocab.stoi[text_field.pad_token]
        self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
    
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
                
    def forward(self, sentences, labels):
        embedded = self.embedding(sentences)

        rnn_out, _ = self.rnn(embedded) ### (max_len, n_sentences, 2*rnn_size).
        
        scores = self.top_layer(rnn_out)  ### (max_len, n_sentences, n_labels).
        
        pad_mask = (sentences == self.pad_word_id).float() ### Find the positions where the token is a dummy padding token.

        # For these positions, we add some large number in the column corresponding to the dummy padding label.
        scores[:, :, self.pad_label_id] += pad_mask*10000
        
        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores = scores.view(-1, self.n_labels)
        labels = labels.view(-1)
        return self.loss(scores, labels)

    def predict(self, sentences):
        embedded = self.embedding(sentences)
        rnn_out, _ = self.rnn(embedded) ### (max_len, n_sentences, 2*rnn_size).
        scores = self.top_layer(rnn_out)  ### (max_len, n_sentences, n_labels).
        pad_mask = (sentences == self.pad_word_id).float() ### Find the positions where the token is a dummy padding token.
        # For these positions, we add some large number in the column corresponding to the dummy padding label.
        scores[:, :, self.pad_label_id] += pad_mask*10000
        # Select the top-scoring labels. The shape is now (max_len, n_sentences).

        predicted = scores.argmax(dim=2)
        # We transpose the prediction to (n_sentences, max_len), and convert it to a NumPy matrix.
        return predicted.t().cpu().numpy(), scores.view(-1, scores.size(-1)).cpu().numpy()






### Extension. RNN + CRF. Very classic on machine translation, bio taging, ner, semantic role labeling, etc.
from torchcrf import CRF

class RNNCRFTagger(nn.Module):
    
    def __init__(self, text_field, label_field, emb_dim, rnn_size, update_pretrained=False):
        super().__init__()
        
        voc_size = len(text_field.vocab)
        self.n_labels = len(label_field.vocab)       
        
        self.embedding = nn.Embedding(voc_size, emb_dim)
        if text_field.vocab.vectors is not None:
            self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
                                                       requires_grad=update_pretrained)

        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, 
                          bidirectional=True, num_layers=1)

        self.top_layer = nn.Linear(2*rnn_size, self.n_labels)
 
        self.pad_word_id = text_field.vocab.stoi[text_field.pad_token]
        self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
    
        self.crf = CRF(self.n_labels)
        
    def compute_outputs(self, sentences):
        embedded = self.embedding(sentences)
        rnn_out, _ = self.rnn(embedded)
        out = self.top_layer(rnn_out)
        
        pad_mask = (sentences == self.pad_word_id).float()
        out[:, :, self.pad_label_id] += pad_mask*10000
        
        return out
                
    def forward(self, sentences, labels):
        # Compute the outputs of the lower layers, which will be used as emission
        # scores for the CRF.
        scores = self.compute_outputs(sentences)

        # We return the loss value. The CRF returns the log likelihood, but we return 
        # the *negative* log likelihood as the loss value.            
        # PyTorch's optimizers *minimize* the loss, while we want to *maximize* the
        # log likelihood.
        return -self.crf(scores, labels)
            
    def predict(self, sentences):
        # Compute the emission scores, as above.
        scores = self.compute_outputs(sentences)

        # Apply the Viterbi algorithm to get the predictions. This implementation returns
        # the result as a list of lists (not a tensor), corresponding to a matrix
        # of shape (n_sentences, max_len).
        return self.crf.decode(scores)