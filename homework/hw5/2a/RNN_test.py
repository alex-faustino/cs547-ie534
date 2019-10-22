import time
import os
import sys
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from RNN_model import RNN_model

#imdb_dictionary = np.load('preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000

x_test = []
with io.open('preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = torch.load('rnn.model')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

opt = 'sgd'
LR = 0.1
# opt = 'adam'
# LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

batch_size = 200
no_of_epochs = 10
L_Y_test = len(y_test)

test_loss = []
test_accu = []

for epoch in range(no_of_epochs):
    # test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = (epoch + 1)*50
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = torch.LongTensor(x_input).to(device)
        target = torch.FloatTensor(y_input).to(device)

        with torch.no_grad():
            loss, pred = model(data,target)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().to("cpu").data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_loss.append(epoch_loss)
    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print(sequence_length, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

data = [test_loss, test_accu]
data = np.asarray(data)
np.save('data.npy',data)