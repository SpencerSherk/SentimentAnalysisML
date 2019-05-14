

# NLP_final
final project for Natural Language Processing, NYU spring 2019, Professor Adam Meyers

TASK LIST:

	IMPLEMENT EMOTIONAL ANALYSIS MODEL - finished (see main.py, using test.json as training/test data)
		- trained using kaggle dataset (https://www.kaggle.com/c/sa-emotions/leaderboard)
		- TODO: change model to allow for input without labels (won't be hard, will do when scraper is working)
		- TODO: optimize model, at 33% accuracy right now (not bad for our 9-sentiment classification task, but might be able to improve)
	  
	IMPLEMENT DATA SCRAPER FOR TWITTER DATA CONTAINING GIVEN WORD
		- generate json file with relevant tweets

	MANUAL DATASET ANNOTATION
		- each of us have to manually annotate the same 100 phrase dataset, compare results, see what human accuracy of manual annotation is (record % agreement) compare to our system to see how it compares 

	IF WE HAVE TIME: (these are all just ideas)
		- create deep learning model as well, or other types of models, compare results
		- link prediction model with datascraper, so user can enter a given word and have the program output the percentage of tweets with each corresponding emotion
		- for manual dataset analysis: score against adjudicated data, or kappa (see lecture notes)
		- maybe external appliactions of our system (less important, during office hrs meyers said he prefers internally-focused projects, but could be easy to do) 

	WRITING THE PAPER:
		- why we chose to analyze emotion using our parameters (we used 9 sentiments: empty, sadness, worry, neutral, surprise, love, fun, hate, happiness) - it's kind of a standard in NLP & related to many papers... will be easy
		- model validation (compare to our manual results)
		- how we implemented our model
		- error analysis 
		- cite papers