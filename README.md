# Dialogue Act Tagger - Applied NLP
Used pyCRF to create a dialog act tagger using the SWBD dataset.

### Data description
The raw data for each utterance in the conversation consists of the speaker name, the tokens and their part of speech tags.

The Switchboard (SWBD) corpus was collected from volunteers and consists of two person telephone conversations about predetermined topics such as child care. SWBD DAMSL refers to a set of dialogue act annotations made to this data. [ This (lengthy) annotation manual ](https://web.stanford.edu/~jurafsky/ws97/manual.august1.html) defines what these dialogue acts mean. 

Individual conversations are stored as individual CSV files. These CSV files have four columns and each row represents a single utterance in the conversation. The order of the utterances is the same order in which they were spoken. The columns are:
* **act_tag** - the dialogue act associated with this utterance. Note, this will be blank for the unlabeled test data we use to test your code.
* **speaker** - the speaker of the utterance (A or B).
* **pos** - a whitespace-separated list where each item is a token, "/", and a part of speech tag (e.g., "What/WP are/VBP your/PRP$ favorite/JJ programs/NNS ?/."). When the utterance has no words (e.g., the transcriber is describing some kind of noise), the pos column may be blank, consist solely of "./.", have a pos but no token, or have an invented token such as MUMBLEx. You can view the text column to see the original transcription.
* **text** - The transcript of the utterance with some cleanup but mostly unprocessed and untokenized. This column may or may not be a useful source of features when the utterance solely consists of some kind of noise.


### Requirements 
Install [pycrfsuite](https://pypi.python.org/pypi/python-crfsuite), a Python interface to [CRFsuite](http://www.chokkan.org/software/crfsuite/). 


### Program description 
It consists of 2 models - a baseline model and an advanced model. 

#### Baseline Tagger
The baseline model is a Python program (baseline_tagger.py) that reads in a directory of CSV files (INPUTDIR), trains a CRFsuite model, tags the CSV files in (TESTDIR), and prints the output labels to OUTPUTFILE. It uses the following as input features for each utterance:
* a feature for whether or not the speaker has changed in comparison with the previous utterance.
* a feature marking the first utterance of the dialogue.
* a feature for every token in the utterance (see the description of CRFsuite for an
example).
* a feature for every part of speech tag in the utterance (e.g., POS_PRP POS_RB POS_VBP POS_.).

Use `python3 baseline_tagger.py INPUTDIR TESTDIR OUTPUTFILE` to run it. 
It will generate a baseline_model file, which is then used by the test function to use that model on test data, which in turn generates an output file. 

#### Advanced Tagger
It includes the following features ina ddition to the baseline feature set: 
In addition to these, the following features were added:
* If the speaker has not changed or is the same
* The length of the utterance in a string format
* Whether that utterance is a question or not
* If it is the last utterance.
* Whether the utterance contains a Mumble or any incoherent text.

Use `python3 advanced_tagger.py INPUTDIR TESTDIR OUTPUTFILE` to run it. 
It will generate a advanced_model file, which is then used by the test function to use that model on test data, which in turn generates an output file. 

#### Analysis/Results: 
The Report file discusses in detail the results and the analysis of these models. 
