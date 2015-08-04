#!/usr/bin/python

import string

class Reader(object):
	""""""

	def __init__(self, pathToText, seqSize, stepSize):
		""""""
		self.seqSize = seqSize
		self.stepSize = stepSize
		self.pathToText = pathToText
		self.sequences = []
		self.targets = []
		self.dict = {}
		self.inv_dict = {}

		self.get_sequences()
		self.build_dict()

	def get_sequences(self):
		""""""
		with open(self.pathToText, 'r') as fstream:
			for line in fstream:
				line = line.replace('\n','') 
				line = list(line)
				currentSequence = []
				for char in line:
					if char in string.printable:
						currentSequence.append(char)
				if currentSequence < self.seqSize:
					continue
				i = 0
				while i+self.seqSize+1 < len(currentSequence):
					self.sequences.append(currentSequence[i:i+self.seqSize])
					self.targets.append(currentSequence[i+1:i+self.seqSize+1])
					i += self.stepSize 
		print "\nNumber of sequences:", len(self.sequences)

	def build_dict(self):
		""""""
		for s in self.sequences:
			for char in s:
				if char not in self.dict:
					self.inv_dict[len(self.dict)] = char
					self.dict[char] = len(self.dict)
		for idx in xrange(len(self.sequences)):
			self.sequences[idx] = [self.dict[char] for char in self.sequences[idx]]
		for idx in xrange(len(self.targets)):
			self.targets[idx] = [self.dict[char] for char in self.targets[idx]]
		print "Vocabulary size:",len(self.dict)
		print ""
		print self.dict.keys()

	def get_vocabulary_size(self):
		""""""
		return len(self.dict)


if __name__ == "__main__":

	r = Reader('./data/lovecraft.txt', 50, 25)