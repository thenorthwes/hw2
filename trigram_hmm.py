import copy
import json
from math import log2, inf

from cache_function import memoize
from hidden_markov_model import hidden_markov_model

END_ = "[E]"
START_ = "[S]"
k_smooth = .0000000001


class trigram_hmm:
    def __init__(self):
        self.tag_sightings: dict = {}  # Count the number of times we see a tag (tag, count)
        self.tag_key_sighting: dict = {} # count the number of times we see a n-1gram
        self.tag_emission_count: dict = {}  # count the number of times a word emitted a tag (word, tag)
        self.word_sighting: dict = {}  # count the number of times we see a word (word)
        self.tag_ngram: dict = {}  # count the number of times we see a tag after a tag ((tag, tag), count)
        self.ngram_size = 3
        self.hmm = hidden_markov_model(2)

    def train(self, training_data_):
        with open(training_data_, "r") as training_data:
            tweet = training_data.readline()
            while tweet:
                self.consume_tweet(tweet)
                tweet = training_data.readline()
        self.hmm.train(training_data_)


    def consume_tweet(self, tweet):
        # add start and stop
        tagged_tweet = json.loads(tweet)
        padded_tweet = [["", START_]] + [["", START_]] + tagged_tweet + [["", END_]]
        for i in range(len(padded_tweet)):
            if i < len(padded_tweet) - (
                    self.ngram_size - 1):  # don't overstep into late sentence words who would include [STOP]
                tagged_word = padded_tweet[i]
                word = tagged_word[0]
                tag = tagged_word[1]
                self.tag_sightings[tag] = self.tag_sightings.get(tag, 0) + 1
                if i > 1:  # skip start
                    t_w_tupe = (tag, word)
                    self.tag_emission_count[t_w_tupe] = self.tag_emission_count.get(t_w_tupe, 0) + 1
                    prev_tags = tuple(x[1] for x in padded_tweet[i - (self.ngram_size - 1):i])
                    self.tag_key_sighting[prev_tags] = self.tag_key_sighting.get(prev_tags,0)+1
                    tag_ngram = prev_tags + (tag,)
                    self.tag_ngram[tag_ngram] = self.tag_ngram.get(tag_ngram, 0) + 1
                    self.word_sighting[word] = self.word_sighting.get(word, 0) + 1

    def emission_prob(self, tag, word):
        loggable_prob = 0
        try:
            loggable_prob = self.tag_emission_count[(tag, word)] / self.tag_sightings[tag]
        except KeyError:
            loggable_prob = k_smooth
        return log2(loggable_prob)

    def transition_prob(self, tag0, tag1, tag2):
        # how many times did tag 1 give us tag 2
        # t2,t1 / t1
        loggable_prob = 0
        try:
            loggable_prob = self.tag_ngram[(tag0, tag1, tag2)] / self.tag_key_sighting[(tag0, tag1)]
        except KeyError:
            loggable_prob = -inf  # Creating unseen tag transitions is risky -- it allows for invalid english
            # That being said -- a language model for emerging languages / patterns (like social media)
            # should probably consider / learn and evolve
        return loggable_prob

class viterbi_tri:
    def __init__(self, hmm: trigram_hmm):
        hmm.emission_prob = memoize(hmm.emission_prob)
        hmm.transition_prob = memoize(hmm.transition_prob)
        self.hmm = hmm
        self.tags = list(hmm.tag_sightings.keys())
        self.table1 = []

    def guess_sentence_tags(self, test_tweet):
        self.table1 = []
        tag_sequence_stack = []
        tagged_tweet = json.loads(test_tweet)
        padded_tweet = [[START_, START_]] + [[START_, START_]] + tagged_tweet + [[END_, END_]]

        for word_tag_probs in padded_tweet:
            self.table1.append([word_tag_probs[0], {}])

        for i in range(len(self.table1)):  # word_tag_probs in self.table1:
            word_tag_probs = self.table1[i]
            word = word_tag_probs[0]
            for tag in self.tags:  ## For every tag this word could be
                emission = self.hmm.emission_prob(tag,
                                                  word)  ## What is the probability of emitting this word as that tag
                prev_tag_ = ""
                mostProbable = -inf
                if word == START_:
                    ## Special start
                    word_tag_probs[1][START_] = (1, prev_tag_)
                else:  # transitions
                    inboundProb = 0
                    for prev_tag in self.tags:  ## For every inbound edge
                        prev_node_prob = self.table1[i - 1][1][prev_tag][0]  # Get the prob to arrive at the prev tag node
                        for prev_prev_tag in self.tags:  ## For every inbound edge of the n-1 node
                            tag_transition_prob = self.hmm.transition_prob(prev_prev_tag, prev_tag, tag)  ## inbound edge

                            prev_prev_node_prob = self.table1[i - 2][1][prev_prev_tag][0]  # Get the prob to arrive at the prev prev node tag
                            probability_of_reaching_node = emission + tag_transition_prob + prev_node_prob + prev_prev_node_prob
                            if probability_of_reaching_node > mostProbable:
                                ## Best transition prob
                                mostProbable = probability_of_reaching_node
                                prev_tag_ = prev_tag
                word_tag_probs[1][tag] = (mostProbable, prev_tag_)

            next_word = self.table1[i + 1][0]
            if next_word == END_:
                # eject and walk backwards
                backpointer = max(word_tag_probs[1].items(), key=lambda point: point[1][0])[0]
                tag_sequence_stack.append(backpointer)
                backindex = i
                while backpointer != '':  ## stop when we get to start
                    backpointer = self.table1[backindex][1][backpointer][1]
                    tag_sequence_stack.append(backpointer)
                    backindex += -1
                tag_sequence_stack.reverse()
                break
        correctly_tagged_words = 0

        for i in range(len(tagged_tweet)):
            hmm_tag = i + 2  ## Skip the special stop and start tags
            if tagged_tweet[i][1] == tag_sequence_stack[hmm_tag]:
                correctly_tagged_words += 1
        print(correctly_tagged_words, len(tagged_tweet))
        return correctly_tagged_words, len(tagged_tweet)
        # go through and calculate the best transition
        # seeking the best way to reach this node from the last set of nodes

    def width(self):
        print(self.tags)
        return len(self.tags)

    def col(self, param):
        return self.table1[0]
