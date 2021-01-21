import math
from unittest import TestCase

from hidden_markov_model import hidden_markov_model, START_, END_, viterbi

test_tweet = "[[\"RT\", \"~\"], [\"@SlimBaeless\", \"@\"], [\":\", \"~\"], [\"I'm\", \"L\"], [\"def\", \"R\"], [\"old\", \"A\"], [\"now\", \"R\"], [\".\", \",\"], [\"I\", \"O\"], [\"really\", \"R\"], [\"can't\", \"V\"], [\"hang\", \"V\"], [\"like\", \"P\"], [\"I\", \"O\"], [\"used\", \"V\"], [\"to\", \"P\"], [\",\", \",\"], [\"it's\", \"L\"], [\"sad\", \"A\"], [\".\", \",\"]]"

veterbi_tweet = "[[\"RT\", \"~\"], [\"@SlimBaeless\", \"@\"]]"
correct_veterbi_guess = "[[\"RT\", \"~\"], [\"@SlimBaeless\", \"@\"]]" ##, [\"@SlimBaeless\", \"@\"], [\":\", \"~\"], [\"I'm\", \"L\"], [\"def\", \"R\"], [\"old\", \"A\"], [\"now\", \"R\"], [\".\", \",\"], [\"I\", \"O\"], [\"really\", \"R\"], [\"can't\", \"V\"], [\"hang\", \"V\"], [\"like\", \"P\"], [\"I\", \"O\"], [\"used\", \"V\"], [\"to\", \"P\"], [\",\", \",\"], [\"it's\", \"L\"], [\"sad\", \"A\"], [\".\", \",\"]]"

class Test(TestCase):
    def test_has_start_once_and_end_0(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        self.assertEqual(1, hmm.tag_sightings[START_])
        self.assertEqual(0, hmm.tag_sightings.get(END_,0))

    def test_consume_tweet_tag_counts(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        self.assertEqual(3, hmm.tag_sightings["R"])
        self.assertEqual(2, hmm.tag_sightings["~"])

    def test_consume_word_tag(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        self.assertEqual(1, hmm.tag_emission_count[("~", "RT")])
        self.assertEqual(2, hmm.tag_emission_count[(",", ".")])

    def test_count_words(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        self.assertEqual(2, hmm.word_sighting["I"])
        self.assertEqual(0, hmm.word_sighting.get("",0))  # validate start and end didnt make it

    def test_consume_tweet_tag_bigrams(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        self.assertEqual(1, hmm.tag_ngram[("~", "@")])
        self.assertEqual(2, hmm.tag_ngram[("V", "P")])
        self.assertEqual(1, hmm.tag_ngram[("[S]", "~")])

    def test_emission_prob(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        self.assertEqual(1, hmm.emission_prob("O", "I"))
        self.assertEqual(0, hmm.emission_prob("O", "hippo"))

    def test_tag_transition_prob(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        self.assertTrue(math.isclose(.6666, hmm.transition_prob("V", "P"), rel_tol=1e-4))
        self.assertTrue(math.isclose(0, hmm.transition_prob("V", ","), rel_tol=1e-4))


    def test_makes_right_sized_viterbi(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        v = viterbi(hmm)
        v.guess_sentence_tags(test_tweet)
        self.assertEqual(10,v.width())

    def test_calcs_first_col_correct(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(test_tweet)
        v = viterbi(hmm)
        v.guess_sentence_tags(test_tweet)
        self.assertEqual([1,0,0,0,0,0,0], v.col(0))


    def test_guesses_right_tag_seq(self):
        hmm = hidden_markov_model(2)
        hmm.consume_tweet(veterbi_tweet)
        v = viterbi(hmm)
        viterbi_guess = v.guess_sentence_tags(veterbi_tweet)
        self.assertEqual(correct_veterbi_guess,viterbi_guess)