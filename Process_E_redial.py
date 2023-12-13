import json
import nltk
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
import torch
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk.download('wordnet')
#nltk.download('omw-1.4')
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(['.', '?', '!', ',', '(', ')' , '[',']'])
from spellchecker import SpellChecker
import re
import threading


lemmatizerLock = threading.Lock()
NUMTHREADS = 20


#Data format
#messages
#text
#movies
#knowledge
#role
#rewrite
#reason
#description
#encourage
#plot
#review
#movieid
#wiki
#conversationID


# Load pre-trained BERT model and tokenizer
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained BART model and tokenizer
tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def main():

    keepGoing = True
    while keepGoing:
        choice = int(input("Enter 1 to process training data, 2 to process test data: "))
        if choice ==1 or choice ==2:
            keepGoing =False
        
    if choice == 1:
        filename= 'C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\E-Redial\\dataset\\eredial.json'
        writeFile = 'E-redial-ALT-TRAINING-LABELS.txt'
    else:
        filename= 'C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\E-Redial\\dataset\\test.json'
        writeFile = 'E-redial-TEST-LABELS.txt'
    
    spell_check =  SpellChecker()
    
    #Open up our file
    with open(filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    #Data already in a list. SO when you loop through it, it is going
    #Through a dictionary of every conversation#Each conversation
    #has 2 main parts messages and conversationId
    #Messages is the supermajority and has many subcategories and will need
    #the most processing. 

    #Containers for data            #Dict Contents
                                    #Key = conversationID

    #Will be used to evaluate each conversation as a whole
    wholeConv = {}                  #{[helperText],[HelperText]}

    #Will be used to evaluate each turn of dialogue
    #A turn being defined as each person having their full say
    #IE Person A says what they want, Person B then responds
    #Once person A speaks again, the turn is over. So a turn may be 
    #Defined as A*B*
    turnDict = {}                   #{[[helper],[seeker]]}

    #Will hold all known movie details
    moviesDict = {}                 #[wiki, plot, review, title]
    #Will hold Conv IDs
    idList = []                     #[conv ID]
    

    forPretrainConv = {}        #Late addition added to ensure order is preserved to proper position embeddings can be added for pretrained model tokenize and stuff
    
    #This will loop through each entry, 1 conversation at a time
    for message in messages:

        #In conjunction, these will record everything that 
        #has been said in a conversation
        totalHelperMssg = []
        totalSeekerMssg = []
        
        preservedOrderList = [] #Will hold conv exactly as it occurs

        #Will hold the turns of a conversation
        turnList = []      #[[[seeker],[helper]]]
        seekerTurnList = []
        helperTurnList = []

        #Will hold Movies
        movieList = []
        #track how many movies mentioned in a conv [for indexing]. 
        movieNum = -1
        #record all movie data
        fullMovieData = []
        

        #Will manage when a turn has occured
        seekerSpoken = False
        helperSpoken = False
        
        
        #Gather the set of messages in the conversation
        workin = message['messages']

        #Get conv ID and add to list
        id = message['conversationId']
        idList.append(id)

        
        
        firstSpeaker = workin[0]['role']

        #Loop through a conversation 
        for dictionary in workin:

            
            
            #Get current sentence    
            sentence = dictionary['text']
            #Determine who is speaking
            role = dictionary['role']
            preservedOrderList.append([role,sentence])
            
            #Manage conversation turn tracking 
            if role == 1:
                totalHelperMssg.append(sentence)
                helperTurnList.append(sentence)
                helperSpoken = True
                
                
            else:
                totalSeekerMssg.append(sentence)
                seekerTurnList.append(sentence)
                seekerSpoken = True

            
            #If both have spoken and a speaker is going again
            #record this turn of the conversation
            if helperSpoken and seekerSpoken and role == firstSpeaker:

                #Add this turn to a list
                turnList.append([helperTurnList,seekerTurnList])

                #Reset turn trackers
                seekerSpoken = False
                helperSpoken = False

                #Reset turn recorder lists
                seekerTurnList = []
                helperTurnList = []
                

            if 'movies' in dictionary.keys() and len(dictionary['movies']) != 0:
                #Remove [] from around the movie title
                for movie in dictionary['movies']:
                    cleanTitle = re.sub("[\\[\\]]", '', movie)
                    
                    movieList.append(cleanTitle)  
                    movieNum +=1  
                        
            if 'knowledge' in dictionary.keys() and len(dictionary['knowledge']) != 0:
                #Subparts to knowledge
                #wiki,plot,review,movieid

                wiki = None
                plot = None
                review = None

                #Get the knowledge dictionary
                
                for i in range(len(dictionary['knowledge'])):
                    
                    curKnowledge = dictionary['knowledge'][i]
                    
                    wiki = curKnowledge['wiki']
                    plot = curKnowledge['plot']
                    review = curKnowledge['review']

                    #Remove extra newlines from the sentences
                    if wiki is not None:
                        wiki = re.sub(r'[\r\n]', ' ', wiki)
                    if plot is not None:
                        plot = re.sub(r'[\r\n]', ' ', plot)
                    if review is not None:    
                        review = re.sub(r'[\r\n]', ' ', review)
                    
                    fullMovieData.append([wiki, plot, review])
                    
                    #description
                    #encourage

        #Add info to conv containers
        wholeConv[hash(id)] = [totalHelperMssg, totalSeekerMssg]
        turnDict[hash(id)] = turnList
        
        #Add movie name to movie data
        for i in range(len(movieList)):
            fullMovieData[i].append(movieList[i])
        #Store full conv data about movies
        moviesDict[hash(id)] = fullMovieData

        #Preserve the order in which the entire conversation was spoken
        forPretrainConv[hash(id)] = preservedOrderList

    
    
    targetScores = addTargetLabels(idList, wholeConv)
    
    
    currFile = open(writeFile, 'w', encoding='utf-8')

    for i in range (len(idList)):
        id = idList[i]
        currFile.write(f'{id},{str(targetScores[hash(id)])}\n')
        

    currFile.close()
    '''
    '''
    keepGoing = True
    while(keepGoing):
        idx = int(input('Enter ID to use'))
        myid = idList[idx]
        print(f'My id is {myid}')
        conv = forPretrainConv[hash(myid)]
        print(conv)
    '''
    '''
    #Proces the dataset with some threads. Start with conversation turn list
    threadList = []
    numTasks = len(idList)//NUMTHREADS
    leftovers = len(idList)%NUMTHREADS
    startNum = 0
    endNum = numTasks
    for i in range(NUMTHREADS):

        #Assign any extra work to the last thread ALWAYS
        if(endNum == len(idList)-leftovers):
            endNum+=leftovers
        thread = threading.Thread(target=tokenizeSpellcheckTurnConv, args = (turnDict, idList, startNum, endNum))
        threadList.append(thread)
        thread.start()
        #Adjust start and end numbers for next thread
        startNum+=numTasks
        endNum += numTasks
        

    #Wait for threads to finish
    for thread in threadList:
        thread.join()
    
    
    #Repeat Process with whole conversation format
    threadList = []
    numTasks = len(idList)//NUMTHREADS
    leftovers = len(idList)%NUMTHREADS
    startNum = 0
    endNum = numTasks
    for i in range(NUMTHREADS):

        #Assign any extra work to the last thread ALWAYS
        if(endNum == len(idList)-leftovers):
            endNum+=leftovers                                   
        thread = threading.Thread(target=cleanFullConv, args = (wholeConv, idList, startNum, endNum))
        threadList.append(thread)
        thread.start()
        #Adjust start and end numbers for next thread
        startNum+=numTasks
        endNum += numTasks
        

    #Wait for threads to finish
    for thread in threadList:
        thread.join()

    
    #Save preprocessed data to file
    saveToFile(idList, turnDict, moviesDict, wholeConv, targetScores)
    '''
    print('Main commented out line 266 bottom')
    
    
#This function will summarize a conversation using Bart, and calculate the cosine similarity using Bert
#It will then score the conversation as good (1) if the cosine similarity is >= .8  and bad (0) if less than that
def addTargetLabels(idList, wholeConv):

    #Threshold for similarity. Adjust if you want higher summary similarity.
    SIMILARITY_THRESHOLD = .87  #pREV WAS .9

    #Loop through every conversation in the list
    targetScores = {} #hash(ID), score 0/1 good/bad
    count = 0
    for id in idList:
        count+=1
        print('Working on id#',count)
        #Get current conversation, split into recommender and seeker (left/right)
        curConv = wholeConv[hash(id)]
        leftString = ' '.join(curConv[0])
        rightString = ' '.join(curConv[1])

        #Make summary of the conversation
        leftSummary = generateSummaryBart(leftString)
        rightSummary = generateSummaryBart(rightString)

        #Calculate the embedding of the summary
        leftEmbedding = calcBertEmbeddings(leftSummary)
        rightEmbedding = calcBertEmbeddings(rightSummary)

        #Calculate the cosine similarity of the 2 summaries
        similarity = cosine_similarity(leftEmbedding, rightEmbedding).item()
        
        # similarity  >= .87 is good (1) anything else is bad (0)
        if similarity >= SIMILARITY_THRESHOLD:
            targetScores[hash(id)] = 1
        else:
            targetScores[hash(id)] = 0

    #return target scores
    return targetScores


#This function generate summaries using BART. 
def generateSummaryBart(text):
    inputs = tokenizer_bart(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_bart.generate(**inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
    return summary

#This  Function calculates the BERT embeddings for entire summarized text
#Apparently BART is better at summarizing but BERT is better for embeddings So we're mixing the 2. IDK this is where we are at right now...
def calcBertEmbeddings(text):
    tokens = tokenizer_bert(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_bert(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


#This function accepts a conversation broken into turns and returns a cleaned version of the data
#Inputs:Dictionary to spell check and tokenize, broken down into conversational turns, a list of conversation id's, start index, stop index
#Outputs: A tokenized, lowercased, spell checked version of the conversation. 
def tokenizeSpellcheckTurnConv(turnDict, idList, startIndex, stopIndex):
    print('Probably need to change tokkSpllChkTurnConv to match with fullconv. Line 251')
    for i in range(startIndex, stopIndex): #len(idList)
        curId = idList[i]
        leftList = []
        rightList= []
        tempList = []
        

        #Each turn is guarenteed to have 2 elements in the array
        #The number of turns per conversation is not guarenteed
        for turn in turnDict[hash(curId)]:
            
            #Turn the words in each array to lowercase versions
            leftBit = list(map(str.lower, turn[0]))
            rightBit = list(map(str.lower, turn[1]))

            #Turn the arrays into strings for tokenization
            for leftSentence, rightSentence in zip(leftBit,rightBit):
                #Tokenize Sentences
                leftTokens = nltk.word_tokenize(leftSentence)
                rightTokens = nltk.word_tokenize(rightSentence)
                

                leftList,errNo = spellCheckWord(leftTokens)
                rightList,errNo = spellCheckWord(rightTokens)

                #Remove any occurences of None in the list after spellchecking
                leftList = [word for word in leftList if word is not None]
                rightList = [word for word in rightList if word is not None]

                #Lemmatizng was causing errors so prevent threading issues with a lock during
                #lemmatization
                with lemmatizerLock:
                    lemmatizer = WordNetLemmatizer()
                    #Lemmatize right list
                    for i, word in enumerate(rightList):

                            lemWord = lemmatizer.lemmatize(word)
                            rightList[i] = lemWord

                    
                    #Lemmatize left list
                    for i, word in enumerate(leftList):
                            lemWord = lemmatizer.lemmatize(word)
                            leftList[i] = lemWord
                        
                    #Add in cleaned data to the fixed list.
                    tempList.append([leftList, rightList])

        #Assign cleaned data to turnDict
        turnDict[hash(curId)] = tempList
        
#This function accepts the dataset of conversations split into 2 components per recommendation session: Recommenders portion and seekers portion
#Unlike the tokenizeSpellcheckTurnConv function, these conversations are not broken into conversational turns. It is everything the seeker said in the conversation
#and everything the recommender said in the conversation
#Inputs:Dictionary to spell check and tokenize, broken down into seeker'sportion and recommender's portion, a list of conversation id's, start index, stop index
#Outputs: A tokenized, lowercased, spell checked version of the conversation. 
def cleanFullConv(wholeConv, idList, startIndex, stopIndex):
    for i in range(startIndex, stopIndex): #len(idList)
        curId = idList[i]
        leftList = []
        rightList= []
        tempList = []
        fullLeftList = []
        fullRightList = []
       

        #Each conversation is guarenteed to have 2 elements in the array
        #The number of elements per conversation is not guarenteed
         
        #Turn the words in each array to lowercase versions
        leftBit = list(map(str.lower, wholeConv[hash(curId)][0]))
        rightBit = list(map(str.lower, wholeConv[hash(curId)][1]))
        
        #Work on leftBit
        for leftSentence in leftBit:
            #Tokenize Sentences
            leftTokens = nltk.word_tokenize(leftSentence)
            leftList,errNo = spellCheckWord(leftTokens)

            #Remove any occurences of None in the list after spellchecking
            leftList = [word for word in leftList if word is not None]
            #Lemmatizng was causing errors so prevent threading issues with a lock during
            #lemmatization
            with lemmatizerLock:
                lemmatizer = WordNetLemmatizer()
                #Lemmatize left list
                for i, word in enumerate(leftList):
                        lemWord = lemmatizer.lemmatize(word)
                        leftList[i] = lemWord
            #Keep cleaned conversation as a whole in their lists
            fullLeftList += leftList

        #work on rightBit
        for rightSentence in rightBit:
            #Tokenize Sentences
            rightTokens = nltk.word_tokenize(rightSentence)
            rightList,errNo = spellCheckWord(rightTokens)

            #Remove any occurences of None in the list after spellchecking
            rightList = [word for word in rightList if word is not None]
            
            #Lemmatizng was causing errors so prevent threading issues with a lock during
            #lemmatization
            with lemmatizerLock:
                lemmatizer = WordNetLemmatizer()
                #Lemmatize right list
                for i, word in enumerate(rightList):
                        lemWord = lemmatizer.lemmatize(word)
                        rightList[i] = lemWord
                
            #Keep cleaned conversation as a whole in their lists
            fullRightList += rightList
            
        #Assign cleaned data to wholeconv
        wholeConv[hash(curId)] = [fullLeftList,fullRightList]
        
 

#This function will spell check a word, and add it to a list as long as the
#word is not a stop word.
#Inputs: A List of tokenized words
#Outputs: A list of spellchecked words with stop words removed
def spellCheckWord(tokenizedList):
    
    spell = SpellChecker()
    cleanedList = []
    skipCheck = False
    dateSpot = False
    movieName=''
    movieYear = ''
    errNo = 0
    for word in tokenizedList:
        word.replace('.','') 
        
        #Movies are surrounded by brackets in the dataset.   
        #Skip spell check for Movie titles as they may have strange words / spellings
        if word == '[':
            skipCheck = True
        elif word == ']': #Movie title is over add in title then year
            skipCheck = False
            dateSpot = False
            cleanedList.append(movieName)
            cleanedList.append(movieYear)
            movieName = ''
            movieYear=''
        elif word == '(':
            dateSpot = True
        
            
        #Spell check word
        if skipCheck is True and dateSpot is False and word not in ['[',']','(',')']: #Add to movie title
            movieName+=word + ' '
        elif dateSpot is True and word not in ['(',')']: #Add to movie Year
            movieYear +=word

            
        elif word not in STOP_WORDS and word not in string.punctuation:
            ogWord = word
            word = spell.correction(word)
            if ogWord != word:
                errNo+=1

            cleanedList.append(word)
    
    return cleanedList,errNo


#Save preprocessed data to Files
#Each ID entry of all files should correspond to one another
#So the 1st entry of the Whole conversation file should correspond to that same conversation
#broken into turns as the 1st entry of the turn file, and to the movie information in the movie file.
def saveToFile(idList, turnDict, moviesDict, wholeConv):
    
    #Open Files
    turnFile = open('E-redial-TRAINING-preprocessed-turns.txt', 'w', encoding='utf-8')
    moviesFile = open('E-redial-TRAINING-preprocessed-movies.txt','w', encoding='utf-8')
    convFile = open('E-redial-TRAINING-preprocessed-conversations.txt','w', encoding='utf-8')

    #For All Conversation IDs
    
    for id in idList:
        
        #Write a conversation to a file
        #The whole conversation will be written to a file in this pattern:
        #ID
        #Whole Conversation from the 1st speaker
        #Whole conversation from the second speaker 
        convFile.write(f'{id}\n')      
        for word in wholeConv[hash(id)][0]: #1st speaker
            convFile.write(word + ' ') 
        convFile.write('\n')
        for word in wholeConv[hash(id)][1]: #2nd speaker
            convFile.write(word + ' ') 
        convFile.write('\n')
        
        #The turns will be written to a file in this pattern:
        #ID
        #1st speaker's turn (word word ... to the end of their turn)\tsecond speakers turn (word word word ... to the end of their turn)
        turnFile.write(f'{id}\n')
        #Loop through the full conversation turn by turn
        for turn in turnDict[hash(id)]:
            #Left person's turn
            for word in turn[0]:
                turnFile.write(f'{word} ')
            
            #Seperate turns with a tab
            turnFile.write('\t')
            #right person's turn
            for word in turn[1]:
                turnFile.write(f'{word} ')
            turnFile.write('\n')
            

        #Movie data will be written to a file in this pattern:
        #ID
        #Number of movies in conversation
        #title
        #wiki-data
        #plot
        #review
        moviesFile.write(f'{id}\n')
        moviesFile.write(f'{len(moviesDict[hash(id)])}\n')
        #Loop through each movie mentioned in a conversation 
        for i in moviesDict[hash(id)]:      
            moviesFile.write(f'{i[-1]}\n{i[0]}\n{i[1]}\n{i[2]}\n')

    
    #Close files
    turnFile.close()
    moviesFile.close()
    convFile.close()

#This will embed a sequence
#Needs a list of IDS, and a conversation List in the form [[spkrRole, sentence]...[spkrRole, sentence]]
def embedSequence(idList, convList):
    print('Embed currently only works for 1 sequence. adjust so it does many')

    #Initialize lists to store tokenized and encoded inputs
    input_ids = []
    token_type_ids = []
    position_ids = []

    # Iterate through the conversation
    for i, (speaker, utterance) in enumerate(dat):
        # Tokenize the utterance
        tokens = tokenizer.tokenize(utterance)
        
        # Encode the tokens and get token type ids
        encoded_dict = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=512, pad_to_max_length=True, return_tensors='pt')
        input_id = encoded_dict['input_ids']
        token_type_id = encoded_dict['token_type_ids']
        
        # Add position encodings
        position_id = torch.tensor([i+1]*input_id.size(1))  # Adding 1 to avoid position_id=0 for the special tokens
        position_ids.append(position_id)
        
        # Add speaker encodings
        speaker_id = torch.tensor([speaker]*input_id.size(1))
        
        # Combine everything
        input_ids.append(input_id)
        token_type_ids.append(token_type_id + speaker_id)

    # Concatenate the lists to get tensors
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    position_ids = torch.cat(position_ids, dim=0)



#Call Main function
if __name__ == '__main__':
    main()