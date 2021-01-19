import time

from hidden_markov_model import hidden_markov_model

TWT_TRAIN_JSON = "./files/CSEP517-HW2-Data/twt.train.json"
TWT_DEV_JSON = "./files/CSEP517-HW2-Data/twt.dev.json"
TWT_TEST_JSON = "./files/CSEP517-HW2-Data/twt.test.json"

def part1(training_data):
    hmm = hidden_markov_model(2)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("******* HW 2 Program Starting")
    part1(TWT_TRAIN_JSON)
    print("******* HW 2 Program completed in: {} seconds".format(round(time.process_time(), 2)))


