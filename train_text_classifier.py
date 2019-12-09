import shutil
import tensorflow as tf
import tensorflow_datasets
from transformers import BertTokenizer
import os
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from classifier.model import TFBertForBinarySequenceClassification
from classifier.helpers import convert_examples_to_features
from keras_radam.training import RAdamOptimizer
from config import CLASSIFIER_BATCH_SIZE, CLASSIFIER_LABELS, CLASSIFIER_MAX_LENGTH, CLASSIFIER_OUTPUT_MODE

def load_examples():
    df = pd.read_excel('input/labeled_tweets_hydrated.xlsx').dropna(axis=0, how='any')
    print(f"Training on {len(df)} examples")
    examples = []
    for _, row in df.iterrows():
        assert row['label'] in ('yes', 'no')
        examples.append({
            'id': row['id'],
            'sentence1': row['text'],
            'label': 0 if row['label'] == 'yes' else 1
        })
    return examples

def get_positive_weight(y):
    assert all(label in (0, 1) for label in y)
    positive = sum(y)
    negative = len(y) - positive
    return negative / positive  # 1 / (positive / negative)


if __name__ == '__main__':
    n_epochs = 6
    weight_decay = 0

    # Load dataset, tokenizer, model from pretrained model/vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = TFBertForBinarySequenceClassification.from_pretrained('bert-base-multilingual-cased')

    examples = load_examples()
    train_examples, valid_examples = train_test_split(examples, test_size=0.2, shuffle=True)  # could stratify but need to rework a bit

    # Prepare dataset as a tf.data.Dataset instance
    train_dataset, train_labels = convert_examples_to_features(tokenizer, CLASSIFIER_OUTPUT_MODE, train_examples, max_length=CLASSIFIER_MAX_LENGTH)
    valid_dataset, valid_labels = convert_examples_to_features(tokenizer, CLASSIFIER_OUTPUT_MODE, valid_examples, max_length=CLASSIFIER_MAX_LENGTH)
    
    pos_weight = get_positive_weight(train_labels)

    # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
    train_dataset = train_dataset.shuffle(buffer_size=len(train_labels), reshuffle_each_iteration=True).batch(CLASSIFIER_BATCH_SIZE).repeat(n_epochs)
    valid_dataset = valid_dataset.batch(CLASSIFIER_BATCH_SIZE)

    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
    # optimizer = RAdam(learning_rate=3e-5, decay=weight_decay, epsilon=1e-08, clipnorm=1.0)
    # https://github.com/CyberZHG/keras-radam/blob/master/keras_radam/training.py
    optimizer = RAdamOptimizer(learning_rate=3e-5, epsilon=1e-08, total_steps=n_epochs * len(train_labels) / CLASSIFIER_BATCH_SIZE, warmup_proportion=0.1)

    # can be enabled if Volta GPU or later
    # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    def weighted_binary_crossentropy(weights):
        def w_binary_crossentropy(y_true, y_pred):
            return tf.keras.backend.mean(tf.nn.weighted_cross_entropy_with_logits(
                labels=tf.cast(y_true, tf.float32),
                logits=y_pred,
                pos_weight=weights,
                name=None
            ), axis=-1)
        return w_binary_crossentropy
    
    loss = weighted_binary_crossentropy(pos_weight)

    model.compile(optimizer=optimizer, loss=loss)

    # Train and evaluate using tf.keras.Model.fit()
    model.fit(
        train_dataset,
        epochs=n_epochs,
        steps_per_epoch=len(train_labels) / CLASSIFIER_BATCH_SIZE,
        validation_data=valid_dataset,
        validation_steps=len(valid_labels) / CLASSIFIER_BATCH_SIZE
    )

    y_pred = np.round(tf.math.sigmoid(model.predict(valid_dataset, steps=len(valid_labels) / CLASSIFIER_BATCH_SIZE)))

    report = classification_report(valid_labels, y_pred, output_dict=False, target_names=CLASSIFIER_LABELS)
    
    print(report)
    
    try:
        shutil.rmtree('input/classifier/')
    except FileNotFoundError:
        pass
    os.makedirs('input/classifier/')
    model.save_pretrained('input/classifier/')
    tokenizer.save_pretrained('input/classifier/')