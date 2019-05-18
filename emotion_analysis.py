import sys
import json
import re
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix


# takes json file as input, splits it into training and test set
IN_FILE = sys.argv[1]

# the datascraped json file  
TERM_FILE = sys.argv[2]

# load training dataset into json object
with open("dev.json", 'r') as file_input:
	json_data=json.load(file_input)

# load json into pandas dataframe
data = pd.DataFrame(json_data)

# stemmer & roots (OPTIMIZE: try other stemmers might work better)
stemmer = SnowballStemmer('english')
words = stopwords.words("english")

shorthand = {
	":)" : "I'm happy",
	":(" : "I'm sad",
	">:(": "I'm angry",
	"B4" : "Before",
	"BTW" : "By the way",
	"DYK" : "Do you know",
	"FB" : "Facebook",
	"FFS" : "For Fuck's Sake",
	"FML" : "Fuck my life",
	"FTW" : "For the win",
	"ICYMI" : "In case you missed it",
	"IDK" : "I don't know",
	"IMHO" : "In my humble opinion",
	"IRL" : "In real life",
	"JK" : "Just kidding",
	"KK" : "okay",
	"LMAO" : "Laughing my ass off",
	"LMFAO" : "Laughing my fucking ass off",
	"LMK" : "Let me know",
	"LOL" : "Laugh out loud",
	"MIRL" : "Meet in real life",
	"NBD" : "No big deal",
	"NSFW" : "Not safe for work",
	"OMG" : "Oh my God",
	"OMFG" : "Oh my fucking God",
	"SFW" : "Safe for work",
	"SMDH" : "Shaking my damn head",
	"SMH" : "Shaking my head",
	"SOB" : "Son of a Bitch",
	"STFU" : "Shut the fuck up",
	"TLDR/TL;DR" : "Too long, didn't read",
	"TY" : "Thank you",
	"YOLO" : "You only live once"
}

# ensure every val in dataframe is type str
data["content"] = data["content"].astype(str)


# replace shorthand
for i in data["content"]:
	for key in shorthand:
		if key in i:
			data["content"][i] = i.replace(key, shorthand.get(key))

# normalize data: strip non-letter chars, split text to list of words, stem, convert list back to strings, set to lowercase, fill "cleaned dataframe"
data['cleaned'] = data["content"].apply(lambda x: " ".join([stemmer.stem(i) 
	for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

# use 90% of dataframe as training set, 10% as test set, "cleaned" dataframe = features, "sentiment" dataframe = target vector
train1, test1, train2, test2 = train_test_split(data['cleaned'], data.sentiment, test_size=0.3)

# helper class to enable all models to train from same dataset format
class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


# emotions
emotions = ['empty', 'sadness', 'worry', 'neutral', 'surprise', 'love', 'fun', 'hate', 'happiness', 'enthusiasm','relief','boredom']
emotion_score = [0,0,0,0,0,0,0,0,0,0,0,0] 

# make predictions for input file
with open("hashtag_trump.json", 'r') as file_input:
	json_data_in = json.load(file_input)

in_data = pd.DataFrame(json_data_in)

# replace shorthand
for row in in_data["content"]:
	for key in shorthand:
		if key in i:
			data["content"][i] = i.replace(key, shorthand.get(key))

# clean input data
in_data['cleaned'] = data["content"].apply(lambda x: " ".join([stemmer.stem(i) 
	for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())


# 3 models:

# Support Vector Machine
pipeline1 = Pipeline([

	# create bag of words with TfidfVectorizer, use bigram model, ignore stopwords
	('vect', TfidfVectorizer(ngram_range=(1, 4), stop_words="english", sublinear_tf=True)),
    ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))]) 

# fit model to training data (pipeline applies above steps)
model1 = pipeline1.fit(train1, train2)
vect = model1.named_steps['vect']
clf = model1.named_steps['clf']

# Returns the mean accuracy on the given test data and labels.
print("")
print("Support Vector Machine Accuracy: " + str(model1.score(test1, test2)))



# Random Forest
pipeline2 = Pipeline([
	    ('vect', TfidfVectorizer(ngram_range=(1, 4), stop_words="english", sublinear_tf=True)),
     	('tfidf', TfidfTransformer()),
     	('clf', RandomForestClassifier()) ])

model2 = pipeline2.fit(train1, train2)
vect = model2.named_steps['vect']
tfidf = model2.named_steps['tfidf']
clf = model2.named_steps['clf']

# Returns the mean accuracy on the given test data and labels.
print("")
print("Random Forest Classifier Accuracy:" + str(model2.score(test1, test2)))



# Naive Bayes
pipeline3 = Pipeline([
	    #('vect', CountVectorizer()),
	    ('vect', TfidfVectorizer(ngram_range=(1, 4), stop_words="english", sublinear_tf=True)),
     	('tfidf', TfidfTransformer()),
     	('clf', MultinomialNB())])

model3 = pipeline3.fit(train1, train2) # data, in_data
vect = model3.named_steps['vect']
tfidf = model3.named_steps['tfidf']
clf = model3.named_steps['clf']

# Returns the mean accuracy on the given test data and labels.
print("")
print("Naive Bayes Accuracy: " + str(model3.score(test1, test2)))


# Perform predictions
for row in in_data['cleaned']:
	raw_text = [row]
	svm_sentiment = model1.predict(raw_text)
	forest_sentiment = model2.predict(raw_text)
	bayes_sentiment = model3.predict(raw_text)
	emotion_score[emotions.index(svm_sentiment)]+=1
	emotion_score[emotions.index(forest_sentiment)]+=1
	emotion_score[emotions.index(bayes_sentiment)]+=1

print("how the internet is feeling about Trump right now:")

# output scores
for el in emotions:
	print (el + ": ", end =" ")
	print("%.0f%%" % (100 * (emotion_score[emotions.index(el)]/sum(emotion_score))))

# test string
print("")
print("")
test_str = ['Today is going to be a great day! I am so excited for everything to come.']

svm_predict = model1.predict(test_str)
rf_predict = model2.predict(test_str)
bayes_predict = model3.predict(test_str)

output = [svm_predict,rf_predict,bayes_predict]
out_str = "detected sentiment(s): "
for i in output:
	if output.index(i) > 0:
		out_str += ', '
	out_str += i


# test any string on the model
#print("test string:")
#print("I am so scared of you, why don't you leave?")
#print(out_str)
#print("")


tweets = ["@tiffanylue i know  i was listenin to bad habit earlier and i started freakin at his part =[",
		"Layin n bed with a headache  ughhhh...waitin on your call...",
		"Funeral ceremony...gloomy friday...",
		"wants to hang out with friends SOON!",
		"@dannycastillo We want to trade with someone who has Houston tickets, but no one will.",
		"Re-pinging @ghostridah14: why didn't you go to prom? BC my bf didn't like my friends",
		"I should be sleep, but im not! thinking about an old friend who I want. but he's married now. damn, &amp; he wants me 2! scandalous!",
		"Hmmm. http://www.djhero.com/ is down",
		"@charviray Charlene my love. I miss you",
		"@kelcouch I'm sorry  at least it's Friday?",
		"cant fall asleep",
		"Choked on her retainers",
		"Ugh! I have to beat this stupid song to get to the next  rude!",
		"@BrodyJenner if u watch the hills in london u will realise what tourture it is because were weeks and weeks late  i just watch itonlinelol",
		"Got the news",
		"The storm is here and the electricity is gone",
		"@annarosekerr agreed",
		"So sleepy again and it's not even that late. I fail once again.",
		"@PerezHilton lady gaga tweeted about not being impressed by her video leaking just so you know",
		"How are YOU convinced that I have always wanted you? What signals did I give off...damn I think I just lost another friend",
		"@raaaaaaek oh too bad! I hope it gets better. I've been having sleep issues lately too",
		"Wondering why I'm awake at 7am,writing a new song,plotting my evil secret plots muahahaha...oh damn it,not secret anymore",
		"No Topic Maps talks at the Balisage Markup Conference 2009   Program online at http://tr.im/mL6Z (via @bobdc) #topicmaps",
		"I ate Something I don't know what it is... Why do I keep Telling things about food",
		"so tired and i think i'm definitely going to get an ear infection.  going to bed &quot;early&quot; for once.",
		"On my way home n having 2 deal w underage girls drinking gin on da bus while talking bout keggers......damn i feel old",
		"@IsaacMascote  i'm sorry people are so rude to you, isaac, they should get some manners and know better than to be so lewd!",
		"Damm servers still down  i need to hit 80 before all the koxpers pass me",
		"Fudge.... Just BS'd that whole paper.... So tired.... Ugh I hate school.....  time to sleep!!!!!!!!!!!",
		"I HATE CANCER. I HATE IT I HATE IT I HATE IT.",
		"It is so annoying when she starts typing on her computer in the middle of the night!",
		"@cynthia_123 i cant sleep",
		"I missed the bl***y bus!!!!!!!!",
		"feels strong contractions but wants to go out.  http://plurk.com/p/wxidk",
		"SoCal!  stoked. or maybe not.. tomorrow",
		"Screw you @davidbrussee! I only have 3 weeks...",
		"@ether_radio yeah :S i feel all funny cause i haven't slept enough  i woke my mum up cause i was singing she's not impressed :S you?",
		"I need skott right now",
		"has work this afternoon",
		"@GABBYiSACTiVE Aw you would not unfollow me would you? Then I would cry",
		"mmm much better day... so far! it's still quite early. last day of #uds",
		"@DavidArchie &lt;3 your gonna be the first  twitter ;) cause your amazing lol. come to canada  would do anything to see you perform",
		"just picked up her Blackberry from the middle of the street! Both she and it are crushed!",
		"Why do I have the feeling I should be packing and hitting for SFO around this time of the year? I think I'm missing something...",
		"@creyes middle school and elem. High schools will remain open for those who need credits to graduate. Cali is broken",
		"Bed!!!!!... its time,..... hope i go to school tomorrow, all though i don't feel very well right now",
		"@onscrn Ahh.  ... Well, I was hoping that I could learn some stuff on the way. ... Why not you and I work on separate things but also",
		"I'm having a problem with my photo here in twitter amf!!!...can't see my face!",
		"@jakeboyd, oh noooo!  if i blow a tire you're reaaaally going to have to send up some batman smoke.",
		"wnna take a bath!!!!",
		"Chocolate milk is so much better through a straw. I lack said straw",
		"why am i so tired?",
		"@djmicdamn hey yu lil fucker i textd yu",
		"@Mennard time diff and i've just been wrapped up in day to day stuff so i havent been tweeting. talk soon,must sleep...up in 6hrs",
		"@benballer  no way! damn that sucks B!  are you ok?",
		"sucks not being able to take days off of work or have the money to take the trip  so sad",
		"bed...sorta. today was good, sara has strep thought Angelina does to; i shared a water with her B4 they told me, i will prob get it to",
		"@ramtops the recession. her hotel are restructuring how the accounts are done. adds a bit more pressure in the short term but we'll cope",
		"@lostluna But I got dibs on Sulu...",
		"@maternitytees Aww  Onward and upwards now, yay! Still sad to leave I bet.",
		"@itsgabbith at once haha.  poor aby still gets sore!",
		"diesel yaris... 70mpg  so sad its not available in the US. That'd be awesome.",
		"I want to buy this great album but unfortunately i dont hav enuff funds  its &quot;long time noisy&quot;",
		"@Pokinatcha  in all honesty...pain   blech.",
		"Ok ... the passengers ... no one is alive ... they're all dead ... you just don't know it til the end ... then you cry ...",
		"At home alone with not much to do",
		"@DavidCookLove ia so much! i haven't really been happy with any of cooks choices for singles.",
		"@vincew @stefanyngo  i fell asleep on the beach and didn't put on enough sunscreen  lol",
		"So i think my son might have the flu cause I def. just cleaned up a stanky puke mess  Poor pumkpin",
		"So great to see Oin &amp; Cynthia.  So happy.  Dinner was great, cute little place.  Too bad Oin got sick afterwards.",
		"I cant give @jertronic any bday nudges.",
		"...and all woman who transfer their first impressions (sexual/maternal) onto a less 'threatening' man -- are themselves as weak as 'Him'",
		"Brothers Bloom won't be opening this weekend in El Paso.  I'll just buy Brick and enjoy that until I can watch Brothers Bloom.",
		"says I miss plurking.  http://plurk.com/p/wxion",
		"Bitten to blood by my cat, on my way for a rabies bacterin. Seems 7 shots for 2 months. Never wash my cats at home again, they hate water",
		"I miss Voobys!",
		"@Dancing_Monk Neither are ELP!!",
		"@havingmysay  dude, that is my favorite sandwich place ever. ummm did you take PICTURES?",
		"is sad that shin ae got married...and it wasn't to alex",
		"@shondarhimes Sure you will tweet about this when you're back, but news is abuzz about TR Knight's leaving &quot;confirmed&quot; today.  Muy triste.",
		"@RachelLock22 ohh thursday i have exams.. all day  what about wednesday ?",
		"there was a mix up with my dentist appt this afternoon. so they rescheduled me for tomorrow @ 9am.",
		"@gcrush @nopantsdance i was just thinking about how excited i am for you guys to move, but then i realized how sad i am to see you go.",
		"goooood mooorning people... sun is out.. definitly spring now, we had our first spring hail storm, my car has dimples now..",
		"@artfuldodga I love those 'it'sakey' USB sticks. We only have the 4GB in Australia",
		"fresh prince and sleepy sleeps my nightly routine  gotta go to Dmv early tmrw",
		"dammit! hulu desktop has totally screwed up my ability to talk to a particular port on one of our dev servers. so i can't watch and code",
		"@emmarler i am jealous of your mom talking to @taylorswift13. i want to see you all our twittering is making me miss you",
		"I can't sleep...I keep thinking about the puppy I played with today",
		".. I'm suppposed to be sleep. But i got some much to do. &amp; i got that one part of the song stuck in my head &quot;your a jerk (iknow)&quot;  blaahh",
		"@lepetitagneau what's going on sweetheart?",
		"How can it be so freaking difficult to get a system-wide spellchecker? Shit, I'd settle for an office suite one. Stupid unhelpful Windows",
		"Last one month due to summer, strawberry is not availble in the Chennai markets!",
		"@willxxmobb work at 6am. Gotta go to bed soon",
		"@RobertF3 correct! I ADORE him. I just plucked him up and put him under my arm cuz he was cryin.  All better now! Hahaha",
		"@sweeetnspicy hiii im on my ipod...i cant fall asleep",
		"dont wanna work 11-830 tomorrow  but i get paid",
		"feels sad coz i wasnt able to play with the guys!!!  http://plurk.com/p/wxiux",
		"PrinceCharming",
		"@ cayogial i wanted to come to BZ this summer :/ not so sure anymore... a teacher's life in the summer SUCKS"]

''' 

#code for creating confusion matrices:


for tweet in tweets:
	print(model1.predict([tweet]))

count = 0


model_ts = ['worry', 'sadness', 'worry', 'enthusiasm', 'worry', 'worry', 'worry', 'worry', 'sadness', 'worry', 'worry', 'worry', 'sadness', 'sadness', 'surprise', 'neutral', 'love', 'worry', 'worry', 'sadness', 'worry', 'sadness', 'worry', 'worry', 'worry', 'worry', 'worry', 'worry', 'sadness', 'hate', 'worry', 'worry', 'neutral', 'neutral', 'neutral', 'worry', 'worry', 'worry', 'neutral', 'worry', 'sadness', 'love', 'neutral', 'worry', 'worry', 'worry', 'worry', 'worry', 'worry', 'sadness', 'neutral', 'worry', 'worry', 'neutral', 'hate', 'sadness', 'worry', 'neutral', 'worry', 'sadness', 'worry', 'sadness', 'neutral', 'sadness', 'worry', 'worry', 'worry', 'worry', 'worry', 'worry', 'neutral', 'worry', 'neutral', 'neutral', 'hate', 'sadness', 'worry', 'worry', 'sadness', 'worry', 'sadness', 'neutral', 'sadness', 'worry', 'love', 'worry', 'worry', 'sadness', 'worry', 'sadness', 'neutral', 'hate', 'worry', 'neutral', 'happiness', 'worry', 'sadness', 'neutral', 'worry', 'neutral']

for i in model_ts:
	count+=1

print("HERE IS THE COUNT: " + str(count))
a_tes = ['sad', 'worry', 'sad', 'neutral', 'sad', 'worry', 'neutral', 'sad', 'neutral', 'neutral', 'neutral', 'worry', 'neutral', 'neutral', 'neutral', 'neutral', 'empty', 'neutral', 'empty', 'worry', 'neutral', 'neutral', 'neutral', 'neutral', 'worry', 'fun', 'sad', 'worry', 'hate', 'hate', 'hate', 'neutral', 'worry', 'neutral', 'neutral', 'hate', 'worry', 'empty', 'neutral', 'sad', 'happy', 'happy', 'worry', 'empty', 'neutral', 'worry', 'worry', 'sad', 'worry', 'neutral', 'sad', 'sad', 'hate', 'worry', 'worry', 'sad', 'worry', 'worry', 'neutral', 'neutral', 'worry', 'sad', 'sad', 'sad', 'sad', 'neutral', 'sad', 'fun','worry', 'neutral', 'sad', 'hate', 'neutral', 'neutral', 'worry', 'sad', 'neutral', 'happy', 'sad', 'sad', 'neutral', 'neutral', 'neutral', 'happy', 'love', 'neutral', 'sad', 'sad', 'happy', 'worry', 'love', 'hate', 'neutral', 'neutral', 'love', 'neutral', 'neutral', 'sad', 'neutral', 'sad']

b_tes = ['sad', 'worry', 'sad', 'neutral', 'sad', 'worry', 'neutral', 'sad', 'neutral', 'neutral', 'neutral', 'worry', 'neutral', 'neutral', 'neutral', 'neutral', 'empty', 'neutral', 'empty', 'worry', 'neutral', 'neutral', 'neutral', 'neutral', 'worry', 'fun', 'sad', 'worry', 'hate', 'hate', 'hate', 'neutral', 'worry', 'neutral', 'neutral', 'hate', 'worry', 'empty', 'neutral', 'sad', 'happy', 'happy', 'worry', 'empty', 'neutral', 'worry', 'worry', 'sad', 'worry', 'neutral', 'sad', 'sad', 'hate', 'worry', 'worry', 'sad', 'worry', 'worry', 'neutral', 'neutral', 'worry', 'sad', 'sad', 'sad', 'sad', 'neutral', 'sad', 'fun','worry', 'neutral', 'sad', 'hate', 'neutral', 'neutral', 'worry', 'sad', 'neutral', 'happy', 'sad', 'sad', 'neutral', 'neutral', 'neutral', 'happy', 'love', 'neutral', 'sad', 'sad', 'happy', 'worry', 'love', 'hate', 'neutral', 'neutral', 'love', 'neutral', 'neutral', 'sad', 'neutral', 'sad']

answer_key = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'sadness', 'worry', 'sadness', 'sadness', 'neutral', 'worry', 'sadness', 'sadness', 'surprise', 'sadness', 'love', 'sadness', 'worry', 'sadness', 'worry', 'fun', 'neutral', 'worry', 'sadness', 'worry', 'sadness', 'worry', 'sadness', 'worry', 'hate', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'sadness', 'worry', 'neutral', 'neutral', 'neutral', 'hapiness', 'fun', 'worry', 'worry', 'empty', 'worry', 'worry', 'sadness', 'neutral', 'neutral', 'neutral', 'worry', 'empty', 'neutral', 'neutral', 'worry', 'enthusiasm', 'neutral', 'neutral', 'sadness', 'worry', 'sadness', 'sadness', 'sadness', 'sadness', 'neutral', 'worry', 'sadness', 'worry', 'happiness', 'neutral', 'worry', 'neutral', 'neutral', 'worry', 'neutral', 'neutral', 'happiness', 'worry', 'worry', 'sadness', 'neutral', 'sadness', 'worry', 'sadness', 'worry', 'hate', 'worry', 'worry', 'worry', 'neutral', 'neutral', 'worry', 'neutral', 'love', 'sadness', 'sadness', 'sadness', 'neutral', 'hate']
david=['sadness', 'worry', 'sadness', 'neutral', 'sadness', 'worry', 'neutral', 'sadness', 'neutral', 'neutral', 'neutral', 'worry', 'neutral', 'neutral', 'neutral', 'neutral', 'empty', 'neutral', 'empty', 'worry', 'neutral', 'neutral', 'neutral', 'neutral', 'worry', 'fun', 'sadness', 'worry', 'hate', 'hate', 'hate', 'neutral', 'worry', 'neutral', 'neutral', 'hate', 'worry', 'empty', 'neutral', 'sadness', 'happy', 'happy', 'worry', 'empty', 'neutral', 'worry', 'worry', 'sadness', 'worry', 'neutral', 'sadness', 'sadness', 'hate', 'worry', 'worry', 'sadness', 'worry', 'worry', 'neutral', 'neutral', 'worry', 'sadness', 'sadness', 'sadness', 'sadness', 'neutral', 'sadness', 'fun','worry', 'neutral', 'sadness', 'hate', 'neutral', 'neutral', 'worry', 'sadness', 'neutral', 'happy', 'sadness', 'sadness', 'neutral', 'neutral', 'neutral', 'happy', 'love', 'neutral', 'sadness', 'sadness', 'happy', 'worry', 'love', 'hate', 'neutral', 'neutral', 'love', 'neutral', 'neutral', 'sadness', 'neutral', 'sadness']
#print confusion_matrix(a_tes, model_ts, labels=['empty', 'sadness', 'worry', 'neutral', 'surprise', 'love', 'fun', 'hate', 'happiness', 'enthusiasm','relief','boredom'])

import matplotlib

#Before
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

labels = [" ",'worried']
cm = confusion_matrix(answer_key, david, labels=['empty', 'sadness', 'worry', 'neutral', 'surprise', 'love', 'fun', 'hate', 'happiness', 'enthusiasm','relief','boredom'])
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Annotator 2')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
'''



