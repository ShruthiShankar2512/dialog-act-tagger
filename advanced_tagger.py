import pycrfsuite
import hw2_corpus_tool as hct
import sys
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_all_data(directory_name):
    return list(hct.get_data(directory_name))


def get_train_test_data(train_directory, test_directory):
    train_data_list = get_all_data(train_directory)
    print(len(train_data_list))
    test_data_list = get_all_data(test_directory)
    print(len(test_data_list))
    return train_data_list, test_data_list


def sentiment_scores(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    """print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

    print("Sentence Overall Rated As", end = " ")"""

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        return "POSITIVE"

    elif sentiment_dict['compound'] <= - 0.05 :
        return "NEGATIVE"

    else :
        return "NEUTRAL"


global glob_previous_speaker

#This function takes as input the DialogUtterance, the previous speaker, if available, else none, and the index.
#It returns a feature list using the hwole utterance
def utterance2features(utterance):
    global glob_previous_speaker
    #feature_list = ["SPEAKER_CHANGED", "FIRST_UTTERANCE", [TOKEN_LIST], [POS_LIST]]
    feature_list = []

    if glob_previous_speaker == None:
        index = 0
    #first feature, has the speaker changed?
    elif glob_previous_speaker != None:
        index = 1
        if utterance[1] != glob_previous_speaker:
            feature_list.append("SPEAKER_CHANGED")
        else:
            feature_list.append("SPEAKER_SAME")


    #second feature - is it the first utterance?
    if index == 0:
        feature_list.append("FIRST_UTTERANCE")


    #third and fourth feature - token list and pos list
    pos_utterance = utterance[2]
    if pos_utterance is not None:
        utterance_length = len(pos_utterance)
        feature_list.append(str(utterance_length))
        sentence = ""
        for i in range(utterance_length):
            sentence += " "
            sentence += pos_utterance[i][0]
            if pos_utterance[i][0] == "?":
                feature_list.append("QUESTION")

            feature_list.append("TOKEN_" + pos_utterance[i][0] )
            feature_list.append("POS_" + pos_utterance[i][1] )
            if "MUMBLE" in pos_utterance[i][0] or "XX" in pos_utterance[i][1]:
                feature_list.append("MUMBLE")
        sentence_sentiment = sentiment_scores(sentence)
        feature_list.append(sentence_sentiment)



    else:
        feature_list.append("NO_WORDS")

    glob_previous_speaker = utterance[1]
    return feature_list


def adv_utterance2features(utterance, index, prev_utterance, prev_index):
    global glob_previous_speaker
    if prev_utterance == None:
        return utterance2features(utterance)
    else:
        if prev_index == 0:
            glob_previous_speaker = None
        prev_utterance_features = utterance2features(prev_utterance)
        #prev_utterance_features = [feat for feat in prev_utterance_features if feat[:4] == "TOK_" ]
        prev_utterance_features2 = ["PREV_"+feat for feat in prev_utterance_features]
        curr_utterance_features = utterance2features(utterance)
        return curr_utterance_features + prev_utterance_features2



#takes the DialogUtterance as input and returns its label, which is the first element in the tuple
def utterance2label(utterance):
    return utterance[0]






def get_adv_features_and_labels(data_list):
    feature_list = []
    label_list = []
    #iterate through the dialogs/files
    count=0
    for dialog in data_list:
        count+=1
        #in each file, iterate through the utterances
        global glob_previous_speaker
        dialog_feature_list = []
        dialog_label_list = []
        dialog_length = len(dialog)
        for i in range(len(dialog)):
            if i == 0:
                glob_previous_speaker = None
                dialog_feature_list.append(adv_utterance2features(dialog[i], i, None, i-1 ))
                dialog_label_list.append(utterance2label(dialog[i]))
            else:
                dialog_feature_list.append(adv_utterance2features(dialog[i], i, dialog[i-1], i-1 ))
                dialog_label_list.append(utterance2label(dialog[i]))
                #print(dialog_feature_list)
                #input()
            if i == dialog_length:
                dialog_feature_list.append("LAST_UTTERANCE")
        feature_list.append(dialog_feature_list)
        label_list.append(dialog_label_list)
        if count%10 == 0:
            print(count)
    return feature_list, label_list





def build_model(x_train, y_train):
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
        })

    trainer.train('advanced_model')
    print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])
    print("done training and saving model")


def predict(x_test, y_test):
    tagger = pycrfsuite.Tagger()
    tagger.open('advanced_model')
    y_pred = [tagger.tag(xseq) for xseq in x_test]
    flat_list_true = [item for sublist in y_test for item in sublist]
    flat_list_pred = [item for sublist in y_pred for item in sublist]
    print(accuracy_score(flat_list_true, flat_list_pred, normalize=True, sample_weight=None))
    return y_pred


def print_predictions(y_pred, output_file):
    with open(output_file, 'w') as f:
        for dialog in y_pred:
            for act_tag in dialog:
                f.write("%s\n" % act_tag)
            f.write("\n")



from sklearn.model_selection import KFold

def perform_k_fold(k, x_train, y_train):
    kf = KFold(n_splits = k, shuffle = True)
    scores = []
    for i in range(k):
        result = next(kf.split(x_train), None)
        train_indices = result[0]
        test_indices = result[1]

        x_training_data = [x_train[i] for i in train_indices]
        y_training_data = [y_train[i] for i in train_indices]
        x_testing_data = [x_train[i] for i in test_indices]
        y_testing_data = [y_train[i] for i in test_indices]
        print(len(x_training_data))
        print(len(y_training_data))
        print(len(x_testing_data))
        print(len(y_testing_data))

        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(x_training_data, y_training_data):
            trainer.append(xseq, yseq)

        trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
        })

        print("start training")
        trainer.train('advanced_model')
        print("finish training ")
        print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])

        tagger = pycrfsuite.Tagger()
        tagger.open('advanced_model')


        y_pred = [tagger.tag(xseq) for xseq in x_testing_data]

        flat_list_true = [item for sublist in y_testing_data for item in sublist]
        flat_list_pred = [item for sublist in y_pred for item in sublist]

        acc_score = accuracy_score(flat_list_true, flat_list_pred, normalize=True, sample_weight=None)
        print(acc_score)
        scores.append(acc_score)
    print(scores)


def main():
    train_directory = sys.argv[1]
    test_directory = sys.argv[2]
    output_file = sys.argv[3]

    train_data_list, test_data_list = get_train_test_data(train_directory, test_directory)
    x_train, y_train = get_adv_features_and_labels(train_data_list)
    print(len(x_train))
    print(len(y_train))
    x_test, y_test = get_adv_features_and_labels(test_data_list)
    print(len(x_test))
    print(len(y_test))


    build_model(x_train, y_train)
    predicted_tags = predict(x_test, y_test)
    print_predictions(predicted_tags, output_file)

    x_train = x_train+x_test
    y_train = y_train+y_test
    perform_k_fold(2, x_train, y_train)


if "__main__":
    main()
