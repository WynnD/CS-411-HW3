# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        labels = self.weights.keys()
        '''
        self.weights is a map from (0-9)->feature->weight
        trainingData is an array of maps from pixel->value
        self.features is an array of pixel coordinates
        '''
        '''
        for each label in trainingLabels, loop through each feature in trainingData
        '''

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                correctLabel = trainingLabels[i]
                data = trainingData[i]
                featureValues = data[0]
                labels = data[1]

                scores = util.Counter()
                for label in labels:
                    scores[label] = featureValues[label] * self.weights

                ans = scores.argMax()

                if ans != correctLabel:
                    self.weights -= featureValues[ans]
                else:
                    self.weights += featureValues[correctLabel]
