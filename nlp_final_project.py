#RUN FROM COMMAND LINE W/ DATASET IN SAME FOLDER

import os
#clears the console; this should work on any os, but at the very least works on mac
os.system('cls' if os.name == 'nt' else 'clear')
print("\n\nLoading...")

#the following need to be installed in the machine running the program
import speech_recognition as sr
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk import corpus
from nltk import WordNetLemmatizer
from datetime import date
import re

#initialize global variables
r = sr.Recognizer()
mic = sr.Microphone()
isFirst = True
transcript = None

question1 = "Tell me a story as if I had been right there next to you. What would I have seen?"
question2 = "Okay, now if I had been there, but I was blind and could only listen. What would I have heard?"
question3 = "Perfect. Now if I were reaching out my hands to feel what was around me, what would I have felt?"
question4 = "I think I'm starting to get the picture. Now if I had been there with you, what would I have smelled?"
question5 = "That's perfect. If I had been there with you, what would I have tasted as well?"
question6 = "I think that gives me a pretty complete picture. Now, to finish off, would you be able to tell me the\
	story again, but this time rewinding from the end?"

#the following are methods to process the text of user responses so that the system can analyze it
def tokenize(text):
    if (not pd.isnull(text)):
        tokens = list(map(lambda x: x.lower(), re.split("\s", text)))
        return tokens
    else:
        return []

def numTokens(text):
    if (text != []):
        return len(text) 
    else: 
        return 0

stopword = corpus.stopwords.words('english')

def remove_stopwords(text):
    if (text != []):
        text = [word for word in text if word not in stopword]
        return text
    else:
        return []

wn = WordNetLemmatizer()

def lemmatizing(text):
    if (text != []):
        text = [wn.lemmatize(word) for word in text]
        return text
    else:
        return []

def getPctUnique(text):
    if (text != []):
        uniq = set(text)
        return (len(uniq) / len(text))
    else:
        return 0

#will return new DataFrame with text, processed text, and analyses
def process_data(full_text):
	processed = pd.DataFrame()
	text = []
	text.append(full_text)
	
	#create and populate columns
	processed['Text'] = text
	processed['Tokenized'] = processed['Text'].apply(lambda x: tokenize(x))
	processed['Processed tokens'] = processed['Tokenized'].apply(lambda x: lemmatizing(remove_stopwords(x)))
	processed['#Tok Total'] = processed['Processed tokens'].apply(lambda x: numTokens(x))
	processed['%Unique w/o Stop'] = processed['Processed tokens'].apply(lambda x: getPctUnique(x))
	processed['#Tok w/o Stop'] = processed['#Tok Total'][0] / 6  #really the average response length
	return processed

#this method listens to audio and writes it to the text file created by user
def get_response():
	global isFirst
	global transcript
	global whichQuestion
	
	#this block will record and identify speech, but loop back if speech not identifiable
	recording = True
	while (recording):
		print("Listening...")
		with mic as source:
			audio = r.listen(source)
		try:
			response_text = r.recognize_google(audio)
		except:
			print('Please try again.\n')
		else:
			recording = False
	
	#write user response to text file
	if (isFirst):
		transcript.write(response_text)
		isFirst = False
	else:
		transcript.write('\n' + response_text)
	
	print("Thank you.")
	time.sleep(3)


def main():
	global transcript
	
	print('\n')
	print("                                                                                          ")
	print("██╗     ██╗███████╗    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ")
	print("██║     ██║██╔════╝    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗")
	print("██║     ██║█████╗      ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝")
	print("██║     ██║██╔══╝      ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗")
	print("███████╗██║███████╗    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║")
	print("╚══════╝╚═╝╚══════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝")
	print("                                                                                          ")
	
	print("Please type your first name and press enter:")
	first_name = input()
	print("\nPlease type your last name and press enter:")
	last_name = input()
	
	#create text file to save responses. Convention: "Last_Name, First_Name Date.txt"
	today = date.today()
	file_name = "%s, %s %s" % (last_name, first_name, str(today))
	transcript = open('%s.txt' % file_name, "a+")
	
	#record transcriptions of audio	
	for i in range(1, 7):
		print('\n\n')
		print(eval('question%d' % i))
		get_response()
	
	print("\n\nListening Finished.")
	print("System thinking...")
	
	#get training data and train model
	#test data come from Boulder Lies and Truth corpus,
	#corpus was adjusted - lies were copied 10 times in order to give the system 
	#a better chance at seeing lies
	test = pd.read_excel('Test_Data1.xlsx')
	X_train = test[['#Tok w/o Stop','%Unique w/o Stop']]
	y_train = test['Rejection']
	logmodel = LogisticRegression(solver='lbfgs')
	logmodel.fit(X_train,y_train)
	
	#close writable document, open again as read only
	transcript.close()
	transcript = open("%s.txt" % file_name, 'r')
	full_text = transcript.read()
	
	#generate data on responses and then use model to predict truthfulness
	to_predict = process_data(full_text)
	X_test = to_predict[['#Tok w/o Stop','%Unique w/o Stop']]
	predictions = logmodel.predict(X_test)
	
	#print out the outcome of analysis - also allow for if there was an issue with the system
	print("\nTHE SYSTEM HAS MADE THE FOLLOWING DETERMINATION:")
	if (predictions[0] == 0):
		print("Your response was: TRUTH\n\n")
	elif (predictions[0] == 1):
		print("Your response was: LIE\n\n")
	else:
		print("Something went wrong here:")
		print(predictions)

if __name__ == '__main__':
    main()