import numpy as np
import matplotlib
from matplotlib import pyplot as plt

with open('to_parse.txt') as f:
    lines = f.readlines()

Finetune_loss = []
Finetune_acc = []
Finetune_acc5 = []

valid_loss = []
valid_acc = []
valid_acc5 = []

for line in lines:
	splitLine = line.split()
	#print(splitLine)
	if len(splitLine) > 0:
		if splitLine[0] == '[Finetune]':
			Finetune_loss.append(splitLine[2].split(',')[0])
			Finetune_acc.append(splitLine[4].split(',')[0])
			Finetune_acc5.append(splitLine[6].split(',')[0])
		if splitLine[0] == '[valid]':
			valid_loss.append(splitLine[2].split(',')[0])
			valid_acc.append(splitLine[4].split(',')[0])
			valid_acc5.append(splitLine[6].split(',')[0])

with open('valid_acc5.txt', 'w') as f:
    for w in valid_acc5:
    	f.write(w)
    	f.write('\n')