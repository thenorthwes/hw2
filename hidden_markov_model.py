import json

END_ = "[END]"
START_ = "[START]"


class hidden_markov_model:
    def __init__(self, ngram_size: int):
        self.tag_sightings: dict = {}  # Count the number of times we see a tag (tag, count)
        self.tag_emission_count: dict = {}  # count the number of times a word emitted a tag (word, tag)
        self.word_sighting: dict = {} # count the number of times we see a word (word)
        self.tag_ngram: dict = {}  # count the number of times we see a tag after a tag ((tag, tag), count)
        self.pos_mapping: dict = {}
        self.ngram_size = ngram_size

    def train(self, training_data):
        with open(training_data, "r") as training_data:
            tweet = training_data.readline()
            while tweet:
                self.consume_tweet(tweet)
                tweet = training_data.readline()

    def consume_tweet(self, tweet):
        # add start and stop
        tagged_tweet = json.loads(tweet)
        padded_tweet = [["", START_]] + tagged_tweet + [["", END_]]
        for i in range(len(padded_tweet)):
            if i < len(padded_tweet) - (self.ngram_size - 1):  # don't overstep into late sentence words who would include [STOP]
                tagged_word = padded_tweet[i]
                word = tagged_word[0]
                tag = tagged_word[1]
                self.tag_sightings[tag] = self.tag_sightings.get(tag, 0) + 1
                if i > 0:  # skip start
                    w_t_tupe = (word, tag)
                    self.tag_emission_count[w_t_tupe] = self.tag_emission_count.get(w_t_tupe, 0) + 1
                    prev_tags = tuple(x[1] for x in padded_tweet[i-(self.ngram_size-1):i])
                    tag_ngram = prev_tags + (tag,)
                    self.tag_ngram[tag_ngram] = self.tag_ngram.get(tag_ngram, 0) + 1
                    self.word_sighting[word] = self.word_sighting.get(word, 0) + 1

    def emission_prob(self, word, tag):
        return self.tag_emission_count[(word,tag)] / self.word_sighting[word]

    def transition_prob(self, tag1, tag2):
        # how many times did tag 1 give us tag 2
        # t2,t1 / t1
        return self.tag_ngram[(tag1,tag2)] / self.tag_sightings[tag1]
