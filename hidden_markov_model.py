import copy
import json
from math import log2, inf

from cache_function import memoize

END_ = "[E]"
START_ = "[S]"
k_smooth = .00000001


class hidden_markov_model:
    def __init__(self, ngram_size: int):
        self.tag_sightings: dict = {}  # Count the number of times we see a tag (tag, count)
        self.tag_emission_count: dict = {}  # count the number of times a word emitted a tag (word, tag)
        self.word_sighting: dict = {}  # count the number of times we see a word (word)
        self.total_word_sighting: int = 0 # count the total number of words
        self.tag_ngram: dict = {}  # count the number of times we see a tag after a tag ((tag, tag), count)
        self.ngram_size = ngram_size
        self.tag_vocab_length: int = 0 # total number of distinct tags

    def train(self, training_data):
        with open(training_data, "r") as training_data:
            tweet = training_data.readline()
            while tweet:
                self.consume_tweet(tweet)
                tweet = training_data.readline()
        self.tag_vocab_length = len(self.tag_sightings.keys())

    def consume_tweet(self, tweet):
        # add start and stop
        tagged_tweet = json.loads(tweet)
        padded_tweet = [["", START_]] + tagged_tweet + [["", END_]]
        for i in range(len(padded_tweet)):
            if i < len(padded_tweet) - (
                    self.ngram_size - 1):  # don't overstep into late sentence words who would include [STOP]
                tagged_word = padded_tweet[i]
                word = tagged_word[0]
                tag = tagged_word[1]
                self.tag_sightings[tag] = self.tag_sightings.get(tag, 0) + 1
                if i > 0:  # skip start
                    t_w_tupe = (tag, word)
                    self.tag_emission_count[t_w_tupe] = self.tag_emission_count.get(t_w_tupe, 0) + 1
                    prev_tags = tuple(x[1] for x in padded_tweet[i - (self.ngram_size - 1):i])
                    tag_ngram = prev_tags + (tag,)
                    self.tag_ngram[tag_ngram] = self.tag_ngram.get(tag_ngram, 0) + 1
                    self.word_sighting[word] = self.word_sighting.get(word, 0) + 1
                    self.total_word_sighting += 1


    def emission_prob(self, tag, word):
        loggable_prob_emission = 0
        try:
            # pretend we saw an unk / tag pair which means 1 more tag sighting for ever tag
            loggable_prob_emission = self.tag_emission_count[(tag, word)] / (self.tag_sightings[tag] + 1)
            # plus tag size for unk
            loggable_prob_word_prob = self.word_sighting[word] / (self.total_word_sighting + self.tag_vocab_length)
        except KeyError:
            # if we haven't seen this word before, assume its an 'UNK'
            # since we pretend we have seen an unk for each tag
            # we have seen unk TAG times
            loggable_prob_word_prob = self.tag_vocab_length / (self.total_word_sighting + self.tag_vocab_length)

        return log2((.98*loggable_prob_emission)+(.02*loggable_prob_word_prob))

    def transition_prob(self, tag1, tag2):
        # how many times did tag 1 give us tag 2
        # t2,t1 / t1
        try:
            return log2(self.tag_ngram[(tag1, tag2)] / self.tag_sightings[tag1])
        except KeyError:
            return -inf  # Creating unseen tag transitions is risky -- it allows for invalid english
            # That being said -- a language model for emerging languages / patterns (like social media)
            # should probably consider / learn and evolve


class viterbi:
    def __init__(self, hmm: hidden_markov_model):
        hmm.emission_prob = memoize(hmm.emission_prob)
        hmm.transition_prob = memoize(hmm.transition_prob)
        self.hmm = hmm
        self.tags = list(hmm.tag_sightings.keys())
        self.table1 = []

    def guess_sentence_tags(self, test_tweet):
        self.table1 = []
        tag_sequence_stack = []
        tagged_tweet = json.loads(test_tweet)
        padded_tweet = [[START_, START_]] + tagged_tweet + [[END_, END_]]

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
                else:  # no transitions
                    inboundProb = 0
                    for prev_tag in self.tags:  ## For every inbound edge
                        tag_transition_prob = self.hmm.transition_prob(prev_tag, tag)  ## inbound edge
                        prev_node_prob = self.table1[i - 1][1][prev_tag][0]  # Get the prob to arrive at the prev tag
                        probability_of_reaching_node = emission + tag_transition_prob + prev_node_prob
                        if probability_of_reaching_node > mostProbable:
                            ## Best transition prob
                            mostProbable = probability_of_reaching_node
                            prev_tag_ = prev_tag
                word_tag_probs[1][tag] = (mostProbable, prev_tag_)

            ## What was the best last tag?

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

        return correctly_tagged_words, len(tagged_tweet)
        # go through and calculate the best transition
        # seeking the best way to reach this node from the last set of nodes

    def width(self):
        print(self.tags)
        return len(self.tags)

    def col(self, param):
        return self.table1[0]
