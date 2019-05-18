#!/usr/bin/python
import os
import sys
import json
import time

TERM=sys.argv[1]
COUNT=sys.argv[2]
NUM_SEARCHES=sys.argv[3]

FINAL_OUTPUT = open(TERM + "_tweets.json", 'w')

def findTweets(numFiles):

    for n in range(int(numFiles)):
    	if TERM[0] == "#":
    		INPUT_JSON = "hashtag_" + TERM[1:] + str(n) + '.txt'
    	else:
        	INPUT_JSON = TERM + str(n) + '.txt'
        OUTPUT_JSON = open(INPUT_JSON, 'w')

        if TERM[0] == "#":
        	command = 'twurl "/1.1/search/tweets.json?q=%23' + TERM[1:].rstrip() + '&count=' + COUNT + '&lang=en" | jq  > ' + INPUT_JSON	
        else:
      	    command = 'twurl "/1.1/search/tweets.json?q=' + TERM + '&result_type=recent&count=' + COUNT + '&lang=en" | jq  > ' + INPUT_JSON

        os.system(command)
        time.sleep(10)

        OUTPUT_JSON.close()

findTweets(NUM_SEARCHES)
