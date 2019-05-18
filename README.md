

# NLP_final
final project for Natural Language Processing, NYU spring 2019, Professor Adam Meyers


Group Roles:

Spencer Sherk - emotion_analysis.py

Anthony - twitter scraper

David - data management & analysis

How to run the system:

 - Run the twitter scraper (see below)
 - run emotion_analysis.py <<test/training set>> <<twitter_data>>

 example run: 

	# fills 4 different text files each with 100 tweets containing "trump" the target word
 > ./findTweets.py trump 100 4 		

 	# converts 4 text files into one json file called "trump.json"
 > python3 findTerms.py trump 4     

 	# run our system using training & test set from "test.json", perform analysis on "trump.json"
 > python3 emotion_analysis.py test.json trump.json 		




How to run Tweet scraper:
- Installing twurl commands:
	- brew install ruby
	- gem install twurl
	- twurl authorize --consumer-key <<key>> --consumer-secret <<secret>>
		- (then enter pin from link that appears)
- Running scraper:
	- ./findTweets.py <<term>> <<number of times to search for tweets>>
	- python3 findTerms.py <<term>> <<number of times tweets were searched>>

