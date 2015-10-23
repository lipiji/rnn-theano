#pylint: skip-file
import sys
import os
import numpy as np

datasets_dir = '/misc/projdata12/info_fil/pjli/data/'
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def char_sequence():
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    #f = open(curr_path + "/shakespeare.txt", "r")
    f = open(curr_path + "/toy.txt", "r")
    for line in f:
        line = line.strip('\n')
        if len(line) < 3:
            continue
        lines.append(line)
        for char in line:
            if char not in w2i:
                i2w[len(w2i)] = char
                w2i[char] = len(w2i)
    f.close()

    for i in range(0, len(lines)):
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = np.float32)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(x)
    print len(w2i)
    return seqs, i2w, w2i
