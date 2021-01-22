import time

from hidden_markov_model import hidden_markov_model, viterbi
from test_hidden_markov_model import test_tweet

TWT_TRAIN_JSON = "./files/CSEP517-HW2-Data/twt.train.json"
TWT_DEV_JSON = "./files/CSEP517-HW2-Data/twt.dev.json"
TWT_TEST_JSON = "./files/CSEP517-HW2-Data/twt.test.json"

def part1(training_data):
    hmm = hidden_markov_model(2)
    hmm.train(training_data)
    v = viterbi(hmm)
    tc = 0
    tt = 0
    with open(TWT_DEV_JSON, "r") as training_data:
        tweet = training_data.readline()
        while tweet:
            correct_total = v.guess_sentence_tags(tweet)
            tc += correct_total[0]
            tt += correct_total[1]
            tweet = training_data.readline()

    print(tc/tt)
    print("Part 1 Complete ")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("******* HW 2 Program Starting")
    part1(TWT_TRAIN_JSON)
    print("******* HW 2 Program completed in: {} seconds".format(round(time.process_time(), 2)))


