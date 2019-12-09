import pickle
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from word_embeddings.config import SEQUENCE_LENGTH, index2label, save_dir, HYDROLOGY_DIM
from word_embeddings.train import predictor
from tensorflow.python.framework import errors_impl


class Classify:
    def __init__(self):
        # Do not use GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        with open('word_embeddings/num_words_flood.json', 'r') as f:
            num_words = json.load(f)

        self.x_text, self.x_hydrology, self.dropout_keep_prob, _, self.predictions, _, _ = predictor(
            num_words, SEQUENCE_LENGTH, HYDROLOGY_DIM)

        self.can_classify = True
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        saver = tf.train.Saver()
        with open('word_embeddings/best_epoch_flood.json', 'r') as f:
            best_model_epoch = json.load(f)
        model_path = os.path.join(save_dir, f'model_epoch_{best_model_epoch}')
        try:
            saver.restore(self.sess, model_path)
        except (errors_impl.NotFoundError, errors_impl.InvalidArgumentError):
            print("Could not load model, not classifying instead")
            self.can_classify = False
        else:
            print("Model restored from file: %s" % model_path)
            with open('word_embeddings/flood_tokenizers.pickle', 'rb') as f:
                self.tokenizers = pickle.load(f)
            with open('word_embeddings/languages_start_flood.json', 'r') as f:
                self.languages_start = json.load(f)

    def is_event_related(self, text, language_code):
        if self.can_classify:
            indices = self.tokenizers[language_code].texts_to_sequences([text])
            indices = np.array(indices)
            text_data = pad_sequences(indices, maxlen=SEQUENCE_LENGTH)
            text_data = text_data + self.languages_start[language_code]
            label_index = self.sess.run([self.predictions], feed_dict={
                                        self.x_text: text_data,
                                        self.x_hydrology: placeholder,
                                        self.dropout_keep_prob: 1
                                        })[0][0]
            label = index2label[label_index]
            return label == 'event'
        else:
            return None

    def are_event_related(self, tuples):
        if self.can_classify:
            data = []
            for text, language_code in tuples:
                indices = self.tokenizers[language_code].texts_to_sequences([text])
                indices = np.array(indices)
                tuple_data = pad_sequences(indices, maxlen=SEQUENCE_LENGTH)
                tuple_data = tuple_data + self.languages_start[language_code]
                data.append(tuple_data[0])
            data = np.array(data)
            label_indices = self.sess.run([self.predictions], feed_dict={
                self.x_text: text_data,
                self.x_hydrology: placeholder,
                self.dropout_keep_prob: 1
                                          })[0]
            return [index2label[index] == 'event' for index in label_indices]
        else:
            return None

if __name__ == '__main__':
    from datetime import datetime
    classifier = Classify()
    text = "UCLA flood: Estimate of gallons lost in main break doubles to 20 million, people evacuated dam breaks http://t.co/bRSaAaSh7n"
    print(classifier.is_event_related(text, 'en'))
    print(classifier.are_event_related([
        (text, 'en'),
        (text, 'en'),
    ]))
    tuples = [(text, 'en') for _ in range(1000)]
    t0 = datetime.now()
    classifier.are_event_related(tuples)

    t1 = datetime.now()
    for text, language_code in tuples:
        classifier.is_event_related(
            text,
            language_code
        )
    t2 = datetime.now()
    print(t1 - t0)
    print(t2 - t1)
