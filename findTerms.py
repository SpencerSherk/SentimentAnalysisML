import sys
import json

TERM=sys.argv[1]
NUM_FILES=sys.argv[2]

FINAL_OUTPUT = open(TERM+'.json', 'w')

def findTweets(numFiles):
    FINAL_OUTPUT.write('[\n')

    for n in range(numFiles):
        TERM_INPUT_FILE_NAME = TERM + str(n) + '.txt'
        TERM_FILE = open(TERM_INPUT_FILE_NAME, 'r')

        tweets = {}

        with open(TERM_INPUT_FILE_NAME) as json_file:
            data = json.load(json_file)

            for i, t in enumerate(data["statuses"]):
                count = 0
                tweet = t.get("text")
                tweets["content"] = tweet
                FINAL_OUTPUT.write('\t')
                json.dump(tweets, FINAL_OUTPUT)
                if n < numFiles and i < len(data["statuses"]) - 1:
                    FINAL_OUTPUT.write(',\n')

        FINAL_OUTPUT.write('\n]')

findTweets(int(NUM_FILES))
