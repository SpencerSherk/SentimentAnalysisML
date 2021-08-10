

# Descriprtion
Natural Language Processing, NYU spring 2019, Advised by Professor Adam Meyers


Group Roles:

Spencer Sherk - Machine Learning Models

Anthony - twitter scraper

David - data management & analysis

Read the paper here : " The Task of MultiTarget Sentiment Analysis "
[NLP_report.pdf](https://github.com/SpencerSherk/SentimentAnalysisML/files/6962207/NLP_report.pdf)

![header](https://user-images.githubusercontent.com/39233109/128880179-1fed3a72-7a70-43f6-91a3-c576cf84c5cf.png)
![p3](https://user-images.githubusercontent.com/39233109/128880197-9957318d-14c9-4cfe-b4fc-9383ce319c45.png)
![p4](https://user-images.githubusercontent.com/39233109/128880201-7a067102-72bd-47aa-ab38-826bf8884bb8.png)
![p5](https://user-images.githubusercontent.com/39233109/128880208-fe5ef35a-c72c-42e9-aa6c-0d63a361b99a.png)
![p6](https://user-images.githubusercontent.com/39233109/128880219-acb58af8-8e79-4075-a1c7-1b35b5adb8ed.png)




# How to run the system:

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

