### author: Chen Zheng
### 10/01/2022

from d2l import torch as d2l
import os
import torch     
from torch import nn     

##########################################################################################################################
# In this task, we choose a famous dataset, Large Movie Review Dataset (IMDB), as the Sentiment Analysis benchmark.
# The data link is shown in here: https://ai.stanford.edu/~amaas/data/sentiment/ .
##########################################################################################################################

##########################################################################################################################
# Set up the hyper-parameters.
print('-----------------------------start set up heper-parameter, load data and glove embedding-----------------------------')
##########################################################################################################################
batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size) 

embed_size = 100
kernel_sizes = [3, 4, 5]
num_channels = [100, 100, 100]
devices = d2l.try_all_gpus()

learning_rate = 1e-3
train_epochs = 5

dropout_rate = 0.5 ## the range from 0 to 1

glove_embedding = d2l.TokenEmbedding("glove.6b.100d")
print('-----------------------------end set up heper-parameter, load data and glove embedding----------------------------')


##########################################################################################################################
# Create the Convolutional Neural Network on Sentiment Analysis Task.
##########################################################################################################################
class Sentiment_Analysis_MLP(nn.Module):                                                     
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):

        super(Sentiment_Analysis_MLP, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentiment_analysis_mlp_1 = nn.Linear(embed_size, embed_size)
        self.sentiment_analysis_mlp_2 = nn.Linear(embed_size, 2)
        self.relu = nn.ReLU()                

    def forward(self, inputs):
        # torch.unsqueeze "adds" a superficial 1 dimension to tensor (at the specified dimension), 
        # while torch.squeeze removes all superficial 1 dimensions from tensor.
        ## embed: (bz, sentence len, emb size) ->  (bz, 1, emb size) -> (bz, emb size)
        embed = torch.mean(self.embedding(inputs), dim=1).squeeze()  
        outputs = self.sentiment_analysis_mlp_1(embed)
        outputs = self.relu(outputs)
        outputs = self.sentiment_analysis_mlp_2(self.dropout(outputs))
        return outputs

##########################################################################################################################
# Initialize the weights.
##########################################################################################################################
def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


##########################################################################################################################
# Instance the MLP and put the Glove embedding into the model
print('-----------------------------start Instance the MLP model-----------------------------')
##########################################################################################################################
embeds = glove_embedding[vocab.idx_to_token]
net = Sentiment_Analysis_MLP(len(vocab), embed_size, kernel_sizes, num_channels)
net.apply(init_weights)
net.embedding.weight.data.copy_(embeds)
print('-----------------------------end Instance the MLP model-----------------------------')

##########################################################################################################################
# Train the model.
print('-----------------------------start training process-----------------------------')
##########################################################################################################################
trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, train_epochs, devices)
print('-----------------------------end training process-----------------------------')

##########################################################################################################################
# Test the model.
print('-----------------------------start testing process-----------------------------')
##########################################################################################################################
sentence = "a very well-made, funny and entertaining picture."
sentence_tensor = torch.tensor(vocab[sentence.split()], device=d2l.try_gpu())

label = torch.argmax(
        net(sentence_tensor.reshape(1, -1))
    )
if label == 1:
    print('The result is positive!')
else:
    print('The result is negative!')