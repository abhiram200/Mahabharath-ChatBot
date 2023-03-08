# Mahabharath-ChatBot

This is a basic chat-bot trained on the basis of Mahabharath.

It's not trained on any advanced dataset. instead trained on some basic queries.

As it doesn't have any big dataset it always asks the user whether the output produced is correct or not and if the user says the output is incorrect then the user can input a correct answer which will be used to generate the output the next time when same or similar query is asked.

Also sometimes you would need to edit the tag name created automatically by the user input (above line). So that the working of the bot is correct.

if you want to modify the dataset according to your own dataset then you can edit the intents.json file

You may also visit this link for creating the intents.json file....     https://github.com/abhiram200/JSON-DataSet-Maker

The above link does not make a dataset based on mahabharath instead it makes the dataset for a normal chat-bot.


# Run the following command to install the necessary python files

git clone https://github.com/abhiram200/Mahabharath-ChatBot.git

cd Mahabharath-ChatBot

pip install -r Requirements.txt

python3 AI_MODEL.py

# Error Fix

If you're facing any errors in the code after doing these then please go to the website https://www.nltk.org/data.html then follow the mannual installation steps to download punkt from tokenizers, omw,omw-1.4, omw_lexicon, and wordnet from corpora.
