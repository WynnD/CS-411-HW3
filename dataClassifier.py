# dataClassifier.py
# -----------------
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


# This file contains feature extraction methods and harness
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import perceptron_pacman
import mira
import samples
import sys
import util
import math
from pacman import GameState, Directions

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def connected(datum):
    """
    Returns 0 if the pixels aren't connected/continuous, and 1 if they are. 
    """
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                left = 0
                right = 0
                top = 0
                bottom = 0
                if x > 0:
                    left = datum.getPixel(x-1, y)
                if x < DIGIT_DATUM_WIDTH-1:
                    right = datum.getPixel(x+1, y)
                if y > 0:
                    bottom = datum.getPixel(x, y-1)
                if y < DIGIT_DATUM_HEIGHT-1:
                    top = datum.getPixel(x, y+1)
                if sum([left, right, top, bottom]) == 0:
                    return 0

    return 1

def connectedWhiteSpace(datum):
    """
    Returns 0 if the white pixels aren't connected/continuous, and 1 if they are. 
    """
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) == 0:
                left = 0
                right = 0
                top = 0
                bottom = 0
                if x > 0:
                    left = datum.getPixel(x-1, y)
                if x < DIGIT_DATUM_WIDTH-1:
                    right = datum.getPixel(x+1, y)
                if y > 0:
                    bottom = datum.getPixel(x, y-1)
                if y < DIGIT_DATUM_HEIGHT-1:
                    top = datum.getPixel(x, y+1)
                directions = [left, right, top, bottom]
                if len(filter(lambda x: x == 0, directions)) == 0:
                    return 0

    return 1

def smallBump(datum):
    bumpSize = 1

    y = DIGIT_DATUM_HEIGHT/2
    x = 0
    while datum.getPixel(x, y) == 0:
        x += 1

    while datum.getPixel(x, y) > 0:
        bumpSize += 1
        x += 1

    toReturn = int(bumpSize < DIGIT_DATUM_WIDTH/3)
    return toReturn


def bigBump(datum):
    bumpSize = 1

    y = DIGIT_DATUM_HEIGHT/2
    x = 0
    while datum.getPixel(x, y) == 0:
        x += 1

    while datum.getPixel(x, y) > 0:
        bumpSize += 1
        x += 1

    toReturn = int(bumpSize >= DIGIT_DATUM_WIDTH/3 and bumpSize <= DIGIT_DATUM_WIDTH/2)
    return toReturn

def veryBigBump(datum):
    bumpSize = 1

    y = DIGIT_DATUM_HEIGHT/2
    x = 0
    while datum.getPixel(x, y) == 0:
        x += 1

    while datum.getPixel(x, y) > 0:
        bumpSize += 1
        x += 1

    toReturn = int(bumpSize > DIGIT_DATUM_WIDTH/2)
    return toReturn

def thin(datum, y):
    width = 1

    x = 0
    while datum.getPixel(x, y) == 0 and x < DIGIT_DATUM_WIDTH-1:
        x += 1

    while datum.getPixel(x, y) > 0 and x < DIGIT_DATUM_WIDTH-1:
        width += 1
        x += 1

    toReturn = int(width <= DIGIT_DATUM_WIDTH/3)
    return toReturn

def wide(datum, y):
    width = 1

    x = 0
    while datum.getPixel(x, y) == 0 and x < DIGIT_DATUM_WIDTH-1:
        x += 1

    while datum.getPixel(x, y) > 0 and x < DIGIT_DATUM_WIDTH-1:
        width += 1
        x += 1

    toReturn = int(width > DIGIT_DATUM_WIDTH/3 and width <= DIGIT_DATUM_WIDTH/2)
    return toReturn

def veryWide(datum, y):
    width = 1

    x = 0
    while datum.getPixel(x, y) == 0 and x < DIGIT_DATUM_WIDTH-1:
        x += 1

    while datum.getPixel(x, y) > 0 and x < DIGIT_DATUM_WIDTH-1:
        width += 1
        x += 1

    toReturn = int(width > DIGIT_DATUM_WIDTH/2)
    return toReturn

def thinBottom(datum, y):
    width = 1

    x = 0
    while datum.getPixel(x, y) == 0 and x < DIGIT_DATUM_WIDTH-1:
        x += 1

    while datum.getPixel(x, y) > 0 and x < DIGIT_DATUM_WIDTH-1:
        width += 1
        x += 1

    toReturn = int(width <= DIGIT_DATUM_WIDTH*.3)
    return toReturn

def wideBottom(datum, y):
    width = 1

    x = 0
    while datum.getPixel(x, y) == 0 and x < DIGIT_DATUM_WIDTH-1:
        x += 1

    while datum.getPixel(x, y) > 0 and x < DIGIT_DATUM_WIDTH-1:
        width += 1
        x += 1

    toReturn = int(width > DIGIT_DATUM_WIDTH*.3 and width <= DIGIT_DATUM_WIDTH*.5)
    return toReturn

def veryWideBottom(datum, y):
    width = 1

    x = 0
    while datum.getPixel(x, y) == 0 and x < DIGIT_DATUM_WIDTH-1:
        x += 1

    while datum.getPixel(x, y) > 0 and x < DIGIT_DATUM_WIDTH-1:
        width += 1
        x += 1

    toReturn = int(width > DIGIT_DATUM_WIDTH*.5)
    return toReturn

def fatLeftHead(datum):
    bumpSize = 1

    y = 0
    x = DIGIT_DATUM_WIDTH/3
    while datum.getPixel(x, y) == 0:
        x += 1

    while datum.getPixel(x, y) > 0:
        bumpSize += 1
        x += 1

    toReturn = int(bumpSize > DIGIT_DATUM_WIDTH/3)
    return toReturn

def fourRowsSixColumns(datum, num_rows, num_cols, row, column):
    row_size = DIGIT_DATUM_HEIGHT/num_rows
    col_size = DIGIT_DATUM_WIDTH/num_cols
    y_start = row * DIGIT_DATUM_HEIGHT/num_rows
    x_start = column * DIGIT_DATUM_WIDTH/num_cols

    num_pixels = row_size * col_size
    active_pixels = 0
    for y in range(y_start, y_start+row_size):
        for x in range(x_start, x_start+col_size):
            if datum.getPixel(x, y) > 0:
                active_pixels += 1
    
    return float(active_pixels)/num_pixels

def column(datum, column):
    col_size = DIGIT_DATUM_WIDTH/8
    x_start = column * DIGIT_DATUM_WIDTH/8

    num_pixels = col_size
    active_pixels = 0
    for y in range(0, DIGIT_DATUM_HEIGHT):
        for x in range(x_start, x_start+col_size):
            if datum.getPixel(x, y) > 0:
                active_pixels += 1
    
    return float(active_pixels)/num_pixels

def row(datum, row):
    row_size = DIGIT_DATUM_HEIGHT/8
    y_start = row * DIGIT_DATUM_HEIGHT/8

    num_pixels = row_size
    active_pixels = 0
    for y in range(0, DIGIT_DATUM_HEIGHT):
        for x in range(y_start, y_start+row_size):
            if datum.getPixel(x, y) > 0:
                active_pixels += 1
    
    return float(active_pixels)/num_pixels

def numVerticalLines(datum):
    n = 0
    for x in range(DIGIT_DATUM_WIDTH):
        num_lines_in_column = 0
        is_line = False
        for y in range(DIGIT_DATUM_HEIGHT):
            pixel = datum.getPixel(x, y) > 0
            if pixel and not is_line:
                num_lines_in_column += 1
                is_line = True
            if not pixel and is_line:
                is_line = False
        if num_lines_in_column > n:
            n = num_lines_in_column

    return n

def numStraightVertLines(datum):
    n = 0
    for x in range(DIGIT_DATUM_WIDTH):
        num_pixels_in_col = 0
        is_line = False
        for y in range(DIGIT_DATUM_HEIGHT):
            pixel = datum.getPixel(x, y) > 0
            if pixel and not is_line:
                num_pixels_in_col += 1
                is_line = True
            elif is_line and not pixel:
                break
            elif is_line:
                num_pixels_in_col += 1
        if num_pixels_in_col > DIGIT_DATUM_HEIGHT*8:
            n += 1

    return n

def numHorizontalLines(datum):
    n = 0
    for y in range(DIGIT_DATUM_HEIGHT):
        num_lines_in_row = 0
        is_line = False
        for x in range(DIGIT_DATUM_WIDTH):
            pixel = datum.getPixel(x, y) > 0
            if pixel and not is_line:
                num_lines_in_row += 1
                is_line = True
            if not pixel and is_line:
                is_line = False
        if num_lines_in_row > n:
            n = num_lines_in_row

    return n

def percentPixels(datum):
    num_pixels = DIGIT_DATUM_HEIGHT * DIGIT_DATUM_WIDTH
    n = 0
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 1:
                n += 1
    
    return float(n)/num_pixels

def numSurfacePoints(datum):
    n = 0
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 1:
                left = 0
                right = 0
                top = 0
                bottom = 0
                top_left = 0
                top_right = 0
                bot_left = 0
                bot_right = 0
                if x > 0:
                    left = datum.getPixel(x-1, y)
                else:
                    n += 1
                    continue
                if x < DIGIT_DATUM_WIDTH-1:
                    right = datum.getPixel(x+1, y)
                else:
                    n += 1
                    continue
                if y > 0:
                    bottom = datum.getPixel(x, y-1)
                else:
                    n += 1
                    continue
                if y < DIGIT_DATUM_HEIGHT-1:
                    top = datum.getPixel(x, y+1)
                else:
                    n += 1
                    continue
                if x > 0 and y > 0:
                    top_left = datum.getPixel(x-1, y-1)
                else:
                    n += 1
                    continue
                if x > 0 and y < DIGIT_DATUM_HEIGHT-1:
                    bot_left = datum.getPixel(x-1, y+1)
                else:
                    n += 1
                    continue
                if x < DIGIT_DATUM_WIDTH-1 and y < DIGIT_DATUM_HEIGHT-1:
                    bot_right = datum.getPixel(x+1, y+1)
                else:
                    n += 1
                    continue
                if x < DIGIT_DATUM_WIDTH-1 and y > 0:
                    top_right = datum.getPixel(x+1, y-1)
                else:
                    n += 1
                    continue
                directions = [left, right, top, bottom]
                if len(filter(lambda x: x == 0, directions)) > 0:
                    n += 1

    return n

def surroundedPixels(datum):
    n = 0
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) == 0:
                top_left = 0
                top_right = 0
                bot_left = 0
                bot_right = 0
                left = 0
                right = 0
                top = 0
                bottom = 0
                if x > 0:
                    left = datum.getPixel(x-1, y)
                if x < DIGIT_DATUM_WIDTH-1:
                    right = datum.getPixel(x+1, y)
                if y > 0:
                    bottom = datum.getPixel(x, y-1)
                if y < DIGIT_DATUM_HEIGHT-1:
                    top = datum.getPixel(x, y+1)
                if x > 0 and y > 0:
                    top_left = datum.getPixel(x-1, y-1)
                if x > 0 and y < DIGIT_DATUM_HEIGHT-1:
                    bot_left = datum.getPixel(x-1, y+1)
                if x < DIGIT_DATUM_WIDTH-1 and y < DIGIT_DATUM_HEIGHT-1:
                    bot_right = datum.getPixel(x+1, y+1)
                if x < DIGIT_DATUM_WIDTH-1 and y > 0:
                    top_right = datum.getPixel(x+1, y-1)
                directions = [left, right, top, bottom, bot_left, bot_right, top_left, top_right]
                if bottom and top:
                    n += 1
                elif left and right:
                    n += 1
                
                '''
                elif len(filter(lambda x: x != 0, directions)) >= 6:
                    n += 1
                ''' 
    return n

def leftStem(datum):
    y = 6
    for x in range(DIGIT_DATUM_WIDTH/2):
        if datum.getPixel(x,y) > 0:
            return 1
    
    return 0

def boxAnalysis(datum):
    left = 0
    right = 0
    top = 0
    bottom = 0
    for i in range(DIGIT_DATUM_WIDTH):
        for j in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(i, j) > 0:
                left = i
            if datum.getPixel(DIGIT_DATUM_WIDTH - i - 1, j) > 0:
                right = DIGIT_DATUM_WIDTH - i - 1

    for j in range(DIGIT_DATUM_WIDTH):
        for i in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(i, j) > 0:
                top = j
            if datum.getPixel(i, DIGIT_DATUM_HEIGHT - j - 1) > 0:
                bottom = DIGIT_DATUM_HEIGHT - j - 1
    
    width = abs(left-right)
    height = abs(top-bottom)
    
    n_intersects = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    for i in range(DIGIT_DATUM_WIDTH):
        if datum.getPixel(i, top) > 0:
            n_intersects['top'] += 1
        if datum.getPixel(i, bottom) > 0:
            n_intersects['bottom'] += 1
    
    for j in range(DIGIT_DATUM_HEIGHT):
        if datum.getPixel(i, left) > 0:
            n_intersects['left'] += 1
        if datum.getPixel(i, right) > 0:
            n_intersects['right'] += 1
    
    toReturn = {'width': width, 'height': height, 'n_intersects': n_intersects}
    return toReturn

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def numHorizontalStems(datum, x):
    n = 0
    if x > -1:
        stem = False
        counter = 0
        for j in range(DIGIT_DATUM_HEIGHT):
            if not stem and datum.getPixel(x, j) > 0:
                stem = True
                counter = 1
            elif stem and datum.getPixel(x, j) == 0:
                n += 1
                stem = False
                counter = 0
            elif stem and counter > 5:
                counter = 0
                stem = False
            elif stem:
                counter += 1

    return n

def numVerticalStems(datum, y):
    n = 0
    if y > -1:
        stem = False
        counter = 0
        for i in range(DIGIT_DATUM_WIDTH):
            if not stem and datum.getPixel(i, y) > 0:
                stem = True
                counter = 1
            elif stem and datum.getPixel(i, y) == 0:
                n += 1
                stem = False
                counter = 0
            elif stem and counter > 5:
                counter = 0
                stem = False
            elif stem:
                counter += 1

    return n

def rowDensity(datum):
    densities = []
    for i in range(DIGIT_DATUM_WIDTH):
        n = 0
        for j in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(i, j) > 0:
                n += 1
        densities.append(float(n)/DIGIT_DATUM_HEIGHT)
    return densities

def colDensity(datum):
    densities = []
    for j in range(DIGIT_DATUM_HEIGHT):
        n = 0
        for i in range(DIGIT_DATUM_WIDTH):
            if datum.getPixel(i, j) > 0:
                n += 1
        densities.append(float(n)/DIGIT_DATUM_HEIGHT)
    return densities

def topStemOnLeft(datum, top):
    y = top + 4
    left = 0
    right = 0
    for j in range(y, y+2):
        for i in range(0, DIGIT_DATUM_WIDTH/2):
            if datum.getPixel(i, j) > 0:
                left += 1

        for i in range(DIGIT_DATUM_WIDTH/2, DIGIT_DATUM_WIDTH):
            if datum.getPixel(i, j) > 0:
                right += 1

    return int(left > right)

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for this datum (datum is of type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractorDigit(datum)
    features['connected'] = connected(datum)
    features['connected_whitespace'] = connectedWhiteSpace(datum)
    features['big_bump'] = veryBigBump(datum)
    features['med_bump'] = bigBump(datum)
    features['sml_bump'] = smallBump(datum)

    features['1vline'] = 0
    features['2vline'] = 0
    features['3vline'] = 0
    features['morethan3vline'] = 0
    features['1hline'] = 0
    features['2hline'] = 0
    features['morethan2hline'] = 0
    features['smallsurface'] = 0
    features['medsurface'] = 0
    features['largesurface'] = 0
    features['leftstem'] = leftStem(datum)
    box_analysis = boxAnalysis(datum)

    top = box_analysis['n_intersects']['top']
    bottom = box_analysis['n_intersects']['bottom']

    features['topStemOnLeft'] = topStemOnLeft(datum, top)
    features['wide_top'] = wide(datum, top)
    features['thin_top'] = thin(datum, top)
    features['vwide_top'] = veryWide(datum, top)
    features['wide_bot'] = wide(datum, bottom)
    features['thin_bot'] = thin(datum, bottom)
    features['vwide_bot'] = veryWide(datum, bottom)

    surrounded = surroundedPixels(datum)
    features['surrounded'] = int(surrounded > 3)

    num_vert_lines = numVerticalLines(datum)
    num_horz_lines = numHorizontalLines(datum)
    surface_points = numSurfacePoints(datum)
    percent_pixels = percentPixels(datum)

    features['%d percent coverage' % round(percent_pixels, 1)*100] = 1
    for i in [x * 0.1 for x in range(0, int(percent_pixels*10))]:
        features['%d percent coverage' % round(i, 1)*100] = 0

    num_vert_straight_lines = numStraightVertLines(datum)

    features['lotsofstraightlines'] = 0
    features['3straightlines'] = 0
    features['2straightlines'] = 0
    features['1straightlines'] = 0
    features['nostraightlines'] = 0

    if num_vert_straight_lines > 3:
        features['lotsofstraightlines'] = 1
    elif num_vert_straight_lines == 3:
        features['3straightlines'] = 1
    elif num_vert_straight_lines == 2:
        features['2straightlines'] = 1
    elif num_vert_straight_lines == 1:
        features['1straightlines'] = 1
    else:
        features['nostraightlines'] = 1

    for i in range(8):
        for j in range(20):
            percent = fourRowsSixColumns(datum, 20, 8, j, i)
            features[('row ' + str(j), 'column ' + str(i))] = int(percent > .15)


    if num_vert_lines == 1:
        features['1vline'] = 1
    elif num_vert_lines == 2:
        features['2vline'] = 1
    elif num_vert_lines == 3:
        features['3vline'] = 1
    else:
        features['morethan3vline'] = 1

    bottom_stems = numVerticalStems(datum, box_analysis['n_intersects']['bottom']+6)
    # print bottom_stems
    features['%d bottom stems' % bottom_stems] = 1
    for i in range(bottom_stems):
        features['%d bottom stems' % bottom_stems] = 0

    top_stems = numVerticalStems(datum, box_analysis['n_intersects']['top'] - 6)
    # print 'top stems ' + str(top_stems)
    features['%d top stems' % top_stems] = 1
    for i in range(top_stems):
        features['%d bottom stems' % top_stems] = 0

    col_density = colDensity(datum)
    row_density = rowDensity(datum)

    for i in range(DIGIT_DATUM_HEIGHT):
        if col_density[i] > .6:
            features['col %d dense' % i] = 1
        else:
            features['col %d dense' % i] = 0
        if row_density[i] > .6:
            features['row %d dense' % i] = 1
        else:
            features['row %d dense' % i] = 0

    return features



def basicFeatureExtractorPacman(state):
    """
    A basic feature extraction function.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """
    features = util.Counter()
    for action in state.getLegalActions():
        successor = state.generateSuccessor(0, action)
        foodCount = successor.getFood().count()
        featureCounter = util.Counter()
        featureCounter['foodCount'] = foodCount
        features[action] = featureCounter
    return features, state.getLegalActions()

def enhancedFeatureExtractorPacman(state):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features
    for each (state, action) pair along with a list of the legal actions

    ##
    """

    features = basicFeatureExtractorPacman(state)[0]
    for action in state.getLegalActions():
        features[action] = util.Counter(features[action], **enhancedPacmanFeatures(state, action))
    return features, state.getLegalActions()

def lose(state):
    return int(state.isLose())

def win(state):
    return int(state.isWin())

def move(action):
    return int(action != Directions.STOP)

def distToNearestGhost(state):
    ghostPositions = state.getGhostPositions()
    pacmanPosition = state.getPacmanPosition()
    if len(ghostPositions) == 0:
        return 0
    else:
        minimum = 10000
        for pos in ghostPositions:
            gx, gy = pos
            px, py = pacmanPosition
            dx = gx - px
            dy = gy - py
            dist = math.sqrt(dy**2 + dx**2)
            if dist < minimum:
                minimum = dist
        return minimum


def distToNearestEdibleGhost(state):
    # use AgentState.scaredTimer!!!
    states = state.getGhostStates()
    if len(states) != 0 and states[0].scaredTimer <= 0:
        return 0
    else:
        return distToNearestGhost(state)


def distToNearestFood(state):
    
    print state

def enhancedPacmanFeatures(state, action):
    """
    For each state, this function is called with each legal action.
    It should return a counter with { <feature name> : <feature value>, ... }
    """
    features = util.Counter()
    successor = state.generatePacmanSuccessor(action)
    features['distNearestGhost'] = distToNearestGhost(successor)
    features['move'] = move(action)
    features['lose'] = lose(successor)
    features['win'] = win(successor)
    distToNearestEdibleGhost(state)
    return features


def contestFeatureExtractorDigit(datum):
    """
    Specify features to use for the minicontest
    """
    features =  basicFeatureExtractorDigit(datum)
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features =  basicFeatureExtractorFace(datum)
    return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the printImage(<list of pixels>) function to visualize features.

    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print "==================================="
            print "Mistake on example %d" % i
            print "Predicted %s; truth is %s" % (prediction, truth)
            print "Image: "
            print rawTestData[i]
            break            


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the
        provided list of pixels.  This will serve as a helper function
        to the analysis function you write.

        Pixels should take the form
        [(2,2), (2, 3), ...]
        where each tuple represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
            # This is so that new features that you could define which
            # which are not of the form of (x,y) will not break
            # this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """


def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces', 'pacman'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-g', '--agentToClone', help=default("Pacman agent to copy"), default=None, type="str")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------"
    print "data:\t\t" + options.data
    print "classifier:\t\t" + options.classifier
    if not options.classifier == 'minicontest':
        print "using enhanced features?:\t" + str(options.features)
    else:
        print "using minicontest feature extractor"
    print "training set size:\t" + str(options.training)
    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        if (options.classifier == 'minicontest'):
            featureFunction = contestFeatureExtractorDigit
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace
    elif(options.data=="pacman"):
        printImage = None
        if (options.features):
            featureFunction = enhancedFeatureExtractorPacman
        else:
            featureFunction = basicFeatureExtractorPacman
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if(options.data=="digits"):
        legalLabels = range(10)
    else:
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == "mostFrequent"):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning for naivebayes"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
    elif(options.classifier == "perceptron"):
        if options.data != 'pacman':
            classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
        else:
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,options.iterations)
    elif(options.classifier == "mira"):
        if options.data != 'pacman':
            classifier = mira.MiraClassifier(legalLabels, options.iterations)
        if (options.autotune):
            print "using automatic tuning for MIRA"
            classifier.automaticTuning = True
        else:
            print "using default C=0.001 for MIRA"
    elif(options.classifier == 'minicontest'):
        import minicontest
        classifier = minicontest.contestClassifier(legalLabels)
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING

        sys.exit(2)

    args['agentToClone'] = options.agentToClone

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

# Dictionary containing full path to .pkl file that contains the agent's training, validation, and testing data.
MAP_AGENT_TO_PATH_OF_SAVED_GAMES = {
    'FoodAgent': ('pacmandata/food_training.pkl','pacmandata/food_validation.pkl','pacmandata/food_test.pkl' ),
    'StopAgent': ('pacmandata/stop_training.pkl','pacmandata/stop_validation.pkl','pacmandata/stop_test.pkl' ),
    'SuicideAgent': ('pacmandata/suicide_training.pkl','pacmandata/suicide_validation.pkl','pacmandata/suicide_test.pkl' ),
    'GoodReflexAgent': ('pacmandata/good_reflex_training.pkl','pacmandata/good_reflex_validation.pkl','pacmandata/good_reflex_test.pkl' ),
    'ContestAgent': ('pacmandata/contest_training.pkl','pacmandata/contest_validation.pkl', 'pacmandata/contest_test.pkl' )
}
# Main harness code



def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']
    
    # Load data
    numTraining = options.training
    numTest = options.test

    if(options.data=="pacman"):
        agentToClone = args.get('agentToClone', None)
        trainingData, validationData, testData = MAP_AGENT_TO_PATH_OF_SAVED_GAMES.get(agentToClone, (None, None, None))
        trainingData = trainingData or args.get('trainingData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][0]
        validationData = validationData or args.get('validationData', False) or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][1]
        testData = testData or MAP_AGENT_TO_PATH_OF_SAVED_GAMES['ContestAgent'][2]
        rawTrainingData, trainingLabels = samples.loadPacmanData(trainingData, numTraining)
        rawValidationData, validationLabels = samples.loadPacmanData(validationData, numTest)
        rawTestData, testLabels = samples.loadPacmanData(testData, numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == "naiveBayes" or options.classifier == "nb"):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)

        print string3
        printImage(features_odds)

    if((options.weights) & (options.classifier == "perceptron")):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] )
    # Run classifier
    runClassifier(args, options)
