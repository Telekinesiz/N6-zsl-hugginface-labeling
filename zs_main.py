import pandas as pd
from ast import literal_eval
from transformers import pipeline


def panda_read(file_name):
    df = pd.read_csv(file_name,
                 sep=',',
                 error_bad_lines=False,
                 engine='python')
    return(df)

def sentiment_classifire(df, column_with_text, candidate_labels):
    classifier = pipeline('zero-shot-classification')
    # available tasks are ['audio-classification', 'automatic-speech-recognition', 'conversational',
    # 'feature-extraction', 'fill-mask', 'image-classification', 'image-segmentation', 'ner', 'object-detection',
    # 'question-answering', 'sentiment-analysis', 'summarization', 'table-question-answering', 'text-classification',
    # 'text-generation', 'text2text-generation', 'token-classification', 'translation', 'zero-shot-classification',
    # 'zero-shot-image-classification', 'translation_XX_to_YY']

    cloumn_with_label = column_with_text + ' label'
    column_index = 1 + int(df.columns.get_loc(column_with_text))
    df[column_with_text] = df[column_with_text].apply(literal_eval)
    df.insert(column_index, cloumn_with_label, 'none')
    limitizer_for_test = 0
    # setting limit for length of input text in order to avoid errors. this model has limit for input
    for ind in df.index:

        stemmed_text = df[column_with_text][ind]
        if len(stemmed_text) >= 212:
            stemmed_text = stemmed_text[0:211]
        else:
            pass


        input_text = ' '.join(str(x) for x in stemmed_text)
        print("row number: " + str(limitizer_for_test))
        print('text: ' + input_text)
        print("length of sentence: " + str(len(input_text)))
        result = classifier(input_text, candidate_labels)

        scores = result['scores']
        labels = result['labels']

        #finding largest score
        largest_score = scores[0]
        for number in scores:
            if number > largest_score:
                largest_score = number

        #finding largest score index and label by undex
        for index, score in enumerate(scores):
            if score == largest_score:
                lab_index = index

        label = labels[lab_index]
        df[cloumn_with_label][ind] = label

        print('Label: '+ label)


        limitizer_for_test  += 1
        #if limitizer_for_test  >= 10:
        #    break

    return (df)



if __name__ == '__main__':


    #specs
    #file_name = 'Ready_dataset.csv'
    file_name = 'Full_data_set_mk13.csv'
    labled_data_file_name = "zs_labled_ready_dataset.csv"
    pd.set_option('display.max_columns', None)

    candidate_labels = ["war", "science", "politics", "economy", "amusement", "hobby"]

    df = panda_read(file_name)

    #df.insert(3, 'Name label', 'none')
    #df.insert(9, 'Comment label', 'none')
    columns_for_classification = ['Comment text','Name']

    for column_with_text in columns_for_classification:
        sentiment_classifire(df,column_with_text,candidate_labels)

    print(df)

    # save to csv
    df.to_csv(labled_data_file_name, index=False)