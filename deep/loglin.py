import torch
import torch.nn.functional as F
from collections import defaultdict
import random
import numpy as np

class NNLM(torch.nn.Module):

    def __init__(self, vocab_len, embedding_dim):
        super(NNLM, self).__init__()
        
        self.embed = torch.nn.Embedding(vocab_len, embedding_dim)
        self.linear = torch.nn.Linear(2*embedding_dim, vocab_len)
        
    def forward(self, inputs):
        word1, word2 = inputs
        embedded = torch.cat((self.embed(word1), self.embed(word2)))
        return F.log_softmax(torch.relu(self.linear(embedded)), dim = 0 )
    
model = NNLM(2, 4)

criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

S = torch.tensor(0)

short_train = [[torch.tensor(1)]]
for t in range(1000):
    optimizer.zero_grad()
    random.shuffle(short_train)
    tot_loss = 0.0
    for i, sentence in enumerate(short_train):
        hist = [S, S]

        for next_word in sentence[:1]+[S]:
            pred = model(hist)

            loss = criterion(pred.view(1,-1), next_word.view(1))
            hist = hist[1:] + [next_word]
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
        

        print t, tot_loss/len(short_train)
