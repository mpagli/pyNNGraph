#!/usr/bin/python

import string
import numpy as np
import cPickle as pkl

class Reader(object):
    """"""

    def __init__(self, pathToText, split):
        """"""
        self.pathToText = pathToText
        self.sequence = []
        self.dict = {}
        self.inv_dict = {}

        self.get_sequence()
        self.build_dict()
        self.split_data(split)

    def get_sequence(self):
        """"""
        with open(self.pathToText, 'r') as fstream:
            for line in fstream:
                line = line.replace('\n','') 
                line = list(line)
                currentSequence = []
                for char in line:
                    if char in string.printable:
                        currentSequence.append(char)
                self.sequence += currentSequence

    def build_dict(self):
        """"""
        self.inv_dict = {i:char for i, char in enumerate(set(self.sequence))}
        self.dict = {char:i for i, char in enumerate(set(self.sequence))}
        self.sequence = [self.dict[char] for char in self.sequence]
        print "Vocabulary size:",len(self.dict)
        print ""
        print self.dict.keys()

    def get_vocabulary_size(self):
        """"""
        return len(self.dict)

    def split_data(self, split):
        """"""
        s = int(split*len(self.sequence))
        self.trainSet = self.sequence[:s]
        self.validSet = self.sequence[s:]

    def save_as(self, fileName):
        """"""
        with open(fileName, 'wb') as outStream:
            pkl.dump(self, outStream, -1)

if __name__ == "__main__":

    r = Reader('./data/war_and_peace.txt', 0.95)
    r.save_as('dataset_WAndP.pkl')