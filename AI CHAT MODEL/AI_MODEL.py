import json
import random
import difflib
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


# Load the intents from the JSON file
with open('intents.json') as file:
    data = json.load(file)


# Create lists of all words, classes, and documents from the intents
words = []
classes = []
documents = []
ignore_chars = ['!', '?', ',', '.']


for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        words.extend(nltk.word_tokenize(pattern))
        # Add the pattern to the documents list along with its class
        documents.append((pattern, intent['tag']))
        # Add the class to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Lemmatize and lower-case each word, and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
words = sorted(list(set(words)))


# Sort classes alphabetically
classes = sorted(classes)


# Create a bag-of-words representation of each document
training_data = []
output_data = []


# Create an empty bag for each document
empty_bag = [0] * len(words)


for document in documents:
    bag = list(empty_bag)
    pattern_words = nltk.word_tokenize(document[0])
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words if word not in ignore_chars]
    # Mark the words that are present in the bag
    for word in pattern_words:
        if word in words:
            bag[words.index(word)] = 1
    training_data.append(bag)
    # Mark the class that the document belongs to
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1
    output_data.append(output_row)


training_data = np.array(training_data)
output_data = np.array(output_data)


# Define and train the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(training_data, output_data, epochs=200, batch_size=5, verbose=1)




# Define a function to predict the class of a given input text
def predict_class(text):
    bag = [0] * len(words)
    pattern_words = nltk.word_tokenize(text)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words if word not in ignore_chars]
    # Mark the words that are present in the bag
    for word in pattern_words:
        if word in words:
            bag[words.index(word)] = 1
    # Predict the class of the input text using the trained model
    result = model.predict(np.array([bag]))[0]
    # Get the class with the highest probability
    max_prob_index = np.argmax(result)
    class_label = classes[max_prob_index]
    # If the probability is below a certain threshold, return None
    if result[max_prob_index] < 0.5:
        return None
    else:
        return class_label



# Choose the best response
def BestResponse(text, responses):
    closest_match = difflib.get_close_matches(text, responses)
    if closest_match:
        return closest_match[0]
    else:
        return random.choice(responses)



# Auto learning from the user


# Load existing data from dataset.json if it exists
try:
    with open("intents.json", "r") as infile:
        data = json.load(infile)
except FileNotFoundError:
    data = {"intents": []}

def add_intent(tag, pattern, response):
    for intent in data["intents"]:
        if intent["tag"] == tag and pattern in intent["patterns"] and response in intent["responses"]:
            print("This intent already exists!")
            return
        elif intent["tag"] == tag:
            intent["patterns"].append(pattern)
            intent["responses"].append(response)
            print("Intent added to existing tag")
            return
    data["intents"].append({
        "tag": tag,
        "patterns": [pattern],
        "responses": [response]
    })
    print("New intent added to dataset")


def TagSelector(question):
    # while True:
        # print("Enter a question (or type 'exit' to quit):")
        # question = input()
    # if question == "exit":
    #     break
    print("Enter your expected answer so that this issue won't come the next time':")
    answer = input()
    
        # Tokenize the question and answer
    tokens = word_tokenize(question.lower())
    tokens.extend(word_tokenize(answer.lower()))

    # Determine the tag name by checking for keywords in the tokens
    if any(word in tokens for word in ["hi", "hello", "hey", "yo", "greetings"]):
        tag = "greeting"
    elif any(word in tokens for word in ["goodbye", "bye", "see you", "ttyl", "talk to you later"]):
        tag = "goodbye"
    elif any(word in tokens for word in ["help", "support", "assistance", "faq", "question"]):
        tag = "help"
    elif any(word in tokens for word in ["narrate a story", "tell a story", "narrate a book", "read a book", "read a story", "read the summary of book", "read the story", "narrate the story", "tell the story"]):
        tag = "story_teller"
    elif any(word in tokens for word in ["fantasy", "magic", "dragons", "sorcery", "wizardry"]):
        tag = "Fantasy"
    elif any(word in tokens for word in ["classic", "literary", "timeless", "time-honored", "ageless"]):
        tag = "classics"
    elif any(word in tokens for word in ["biography", "autobiography", "life story", "memoir", "personal history"]):
        tag = "biography"
    elif any(word in tokens for word in ["history", "past", "historical", "ancestry", "heritage"]):
        tag = "history"
    elif any(word in tokens for word in ["self help", "personal growth", "self improvement", "self development"]):
        tag = "self_help"
    elif any(word in tokens for word in ["thriller", "suspense", "mystery", "sleuth", "detective"]):
        tag = "Thriller"
    elif any(word in tokens for word in ["mystery", "enigma", "puzzle", "secret", "conundrum"]):
        tag = "Mystery"
    elif any(word in tokens for word in ["romance", "love", "passion", "affection", "devotion"]):
        tag = "Romance"
    elif any(word in tokens for word in ["science fiction", "sci-fi", "futuristic", "outer space", "extraterrestrial"]):
        tag = "Science Fiction"
    elif any(word in tokens for word in ["thanks", "thank you", "thx", "appreciate"]):
        tag = "thanks"
    elif any(word in tokens for word in ["joke", "funny", "laugh", "humor"]):
        tag = "joke"
    elif any(word in tokens for word in ["movie", "show", "entertainment", "watch"]):
        tag = "entertainment"
    elif any(word in tokens for word in ["music", "song", "playlist", "genre", "band"]):
        tag = "music"
    elif any(word in tokens for word in ["book", "author", "genre", "reading", "recommendation"]):
        tag = "books"
    elif any(word in tokens for word in ["sports", "game", "score", "team", "athlete"]):
        tag = "sports"
    elif any(word in tokens for word in ["history", "fact", "date", "timeline"]):
        tag = "history"
    elif any(word in tokens for word in ["calculator", "compute", "math", "equation"]):
        tag = "calculator"
    elif any(word in tokens for word in ["health", "nutrition", "fitness", "wellness", "medical"]):
        tag = "health"
    elif any(word in tokens for word in ["chat", "conversation", "talk", "discuss", "speak"]):
        tag = "chat"
    elif any(word in tokens for word in ["birthday", "celebration", "party", "gift"]):
        tag = "celebration"
    elif any(word in tokens for word in ["travel", "destination", "vacation", "tourism"]):
        tag = "travel"
    elif any(word in tokens for word in ["pet", "animal", "dog", "cat", "fish"]):
        tag = "pets"
    elif any(word in tokens for word in ["food", "recipe", "restaurant", "cuisine"]):
        tag = "food"
    elif any(word in tokens for word in ["movie", "cinema", "showtime", "trailer"]):
        tag = "movie"
    elif any(word in tokens for word in ["relationship", "dating", "love", "breakup", "marriage"]):
        tag = "relationship"
    elif any(word in tokens for word in ["technology", "software", "hardware", "device", "internet"]):
        tag = "technology"
    elif any(word in tokens for word in ["fashion", "style", "clothing", "accessories", "beauty"]):
        tag = "fashion"
    elif any(word in tokens for word in ["education", "school", "college", "course", "degree"]):
        tag = "education"
    elif any(word in tokens for word in ["how are you", "how's it going", "how have you been", "how's your day", "what's up"]):
        tag = "how_are_you"
    elif any(word in tokens for word in ["sad", "happy", "angry", "frustrated", "anxious", "excited", "nervous", "stressed", "overwhelmed", "confused", "disappointed", "content", "grateful", "hopeful", "proud", "embarrassed", "surprised", "jealous", "scared"]):
        tag = "feelings"
    elif any(word in tokens for word in ["what is", "who is", "where is", "why", "when", "how", "which", "can you", "do you", "tell me", "explain", "define", "clarify", "elaborate", "could you", "would you"]):
        tag = "question_and_answer"
    else:
        # print("the input does not match any previously defined tags. So please give any tag name manually")
        # tag = input()
        tag = "user_input"

    # If a tag name was found, add the intent to the data dictionary
    if tag:
        add_intent(tag, question, answer)
        print(f"data added with tag: {tag}")

    # Write the updated data back to dataset.json
    with open("intents.json", "w") as outfile:
        json.dump(data, outfile, indent=4, separators=(',', ': '))
    
    return "Data written to my brain"
    




# Define a function to generate a response to a given input text
def generate_response(text):
    # Predict the class of the input text
    class_label = predict_class(text)
    # If the class cannot be predicted, return a default response
    if not class_label:
        print("I'm sorry, I don't understand.")
        # TagSelector(text)
        return TagSelector(text)
    else:
        # Find the appropriate response for the predicted class
        for intent in data['intents']:
            if intent['tag'] == class_label:
                responses = intent['responses']
                if class_label == "user_input":
                    # Look for responses based on order of patterns
                    for i in range(len(intent["patterns"])):
                        if intent["patterns"][i] in text:
                            return responses[i]
                        else:
                            return BestResponse(text, responses)
                else:
                    # Choose the best response based on closest match
                    return BestResponse(text, responses)





while True:
    user_input = input("You: ")
    response = generate_response(user_input)
    print("Chatbot: " + response)
    print("\n\nAs the model is currently under development please provide some feedback")
    print("\nIs the provided answer correct as for you?")
    feedback = input()
    if "no" in feedback:
        print(TagSelector(user_input))
    else:
        pass
