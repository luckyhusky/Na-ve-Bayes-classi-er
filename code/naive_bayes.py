# -*- mode: Python; coding: utf-8 -*-
from classifier import Classifier
from corpus import Document, BlogsCorpus
from random import shuffle, seed
from math import log
import nltk
from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import RSLPStemmer

from nltk.corpus import stopwords

import numpy
import string


class NaiveBayes(Classifier):
    """A naïve Bayes classifier."""

    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)
 
    def get_model(self):
        return self.myModel

    def set_model(self, model):
        self.myModel = model

    model = property(get_model, set_model)

    def train(self, instances):
        """Construct a statistical model from labeled instances."""
        self.myModel = {}
        count = 0
        for instance in instances:
            # Use dictionary for myModel
            if instance.label == "":
                continue
            count += 1
            if instance.label not in self.myModel:
                self.myModel[instance.label] = Label()
            lbl = self.myModel[instance.label]
            lbl.docCnt += 1
            for feature in instance.features():
                lbl.rvRec[feature] = lbl.rvRec.get(feature, 0) + 1
                lbl.rvCnt += 1
        
        for _, lbl in self.myModel.iteritems():
            lbl.log_doc_prob = log(lbl.docCnt) - log(count)

    def classify(self, instance):
        """Classify an instance and return the expected label."""
        result, prob, curprob = None, None, 0

        for label, lbl in self.myModel.iteritems():
            curprob = lbl.log_doc_prob

            for feature in instance.features():
                curprob += lbl.log_feature_prob(feature)

            if prob is None or prob < curprob:
                prob = curprob
                result = label
        return result


class BagOfWords(Document):
    def features(self):
        return self.data.split()


class BagOfWordsStemmed(Document):
    def features(self):
        """Stem the word"""
        st = RSLPStemmer()    
        return [st.stem(word) for word in self.data.split()]


class BagOfWordsTokenized(Document):
    def features(self):
        """Trivially tokenized words."""
        return word_tokenize(self.data)


class BagOfWordsWithoutPunctunation(Document):
    def features(self):
        exclude = set(string.punctuation)
        out = []
        for word in self.data.split():
            out.append(''.join(ch for ch in word if ch not in exclude)) 
        return out


class NGram(Document):
    def features(self):
        """Use N gram to extract feature """
        n = 2
        data = self.data.split()
        st = RSLPStemmer()    
        data = [st.stem(word) for word in self.data.split()]
        out = []
        for i  in range(n, len(self.data.split()) - n + 1):
            out.append(data[i - n:i])
            out.append(data[i + 1:i + n])
        return [' '.join(x) for x in out]


class FMeasure(Document):
    def features(self):
        tagWords = nltk.pos_tag(word_tokenize(self.data))
        tag_fd = nltk.FreqDist(tag for (word, tag) in tagWords)
        return dict((key, value) for key, value in tag_fd.most_common())


class Label(object):
    def __init__(self):
        self.docCnt = 0
        self.rvCnt = 0  # rv for random variable
        self.rvRec = {}

    def log_feature_prob(self, feature):
        freq = self.rvRec.get(feature, 0)
        return log(freq + 0.1) - log(self.rvCnt + 0.1 * len(self.rvRec))


def split_blogs_corpus(document_class, count=3000):
    """Split the blog post corpus into trainin5 and test sets"""
    blogs = BlogsCorpus(document_class=document_class)
    seed(hash("blogs"))
    shuffle(blogs)
    return (blogs[:count], blogs[count:])


def accuracy(classifier, test):
    """Find the performance of model"""
    correct = [classifier.classify(x) == x.label for x in test]
    return float(sum(correct)) / len(correct)


if __name__ == '__main__':
    train, test = split_blogs_corpus(NGram)
    classifier = NaiveBayes()
    classifier.train(train)
    classifier.save("myModel.dat")
    print(accuracy(classifier, test))
