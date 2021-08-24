#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

not_bothering = """i'd
we'd
lmao
lang=
&raquo
6.
they've
😂
they'd
t.co
you've
9)|!(ie)]>
@realdonaldtrump
f.d.a.
&amp
4.
he'd
y'all
<html
we've
7.
you'd
careerarc
8.
i've
:"""
not_bothering = set(not_bothering.split("\n"))

vocab = {}
inverted_vocab = {}
vocab_file = open("idf.txt")

i = 0
for line in vocab_file:
    word, _ = line.split()
    if word in not_bothering:
        continue
    vocab[word] = i
    inverted_vocab[i] = word
    i += 1

print "Loading Glove Vectors"
glove_vecs = {}
vec_file = open('glove.6B.300d.txt')
for i, line in enumerate(vec_file):
    tline = line.strip().split()
    word, glove_vec = tline[0], np.array(tline[1:], dtype=np.float16)
    glove_vecs[word] = glove_vec / np.linalg.norm(glove_vec)


squash = 1
model = torch.nn.Sequential(
    torch.nn.Linear(len(vocab)/squash,300),
)
model = model.cuda()

x = np.identity(len(vocab)/squash)
print len(x)
y = [glove_vecs[inverted_vocab[i]] for i in range(len(vocab)/squash)]

x = torch.tensor(x, device=device, dtype = dtype)
y = torch.tensor(y, device=device, dtype = dtype)
loss_fn = torch.nn.MSELoss(reduction='sum')

print "Training"
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for t in range(50000):
    try:
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        if t%20 == 0:
            print t, loss.item()

        optimizer.zero_grad()
        # my_lambda = torch.tensor(0.5, device=device, dtype = dtype)
        # l1_reg = torch.tensor(0., device=device, dtype = dtype)


        # for param in model.parameters():
        #     l1_reg += torch.sum(torch.abs(param))

        # loss += my_lambda * l1_reg
        loss.backward()

        optimizer.step()
    except:
        print "early breakin"
        break
print t, loss.item() # - (my_lambda * l1_reg).item()