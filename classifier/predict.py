from transformers import BertTokenizer
from classifier.model import TFBertForBinarySequenceClassification
from classifier.helpers import convert_examples_to_features
import numpy as np
import tensorflow as tf

from config import CLASSIFIER_BATCH_SIZE, CLASSIFIER_LABELS, CLASSIFIER_MAX_LENGTH, CLASSIFIER_OUTPUT_MODE


class Predictor:
    def __init__(self):
        self.model = TFBertForBinarySequenceClassification.from_pretrained('input/classifier/')
        self.tokenizer = BertTokenizer.from_pretrained('input/classifier/')

    def __call__(self, example_or_examples):
        if not isinstance(example_or_examples, (list, tuple)):
            examples = [example_or_examples]
            return_as_list = False
        else:
            examples = example_or_examples
            if not examples:
                return []
            return_as_list = True
        dataset, _ = convert_examples_to_features(self.tokenizer, CLASSIFIER_OUTPUT_MODE, examples, max_length=CLASSIFIER_MAX_LENGTH)
        dataset = dataset.batch(CLASSIFIER_BATCH_SIZE)
        logits = self.model.predict(dataset)
        predictions = np.round(tf.math.sigmoid(logits))
        predictions = predictions[:, 0].astype(np.int32).tolist()
        predictions = [CLASSIFIER_LABELS[i] for i in predictions]
        if return_as_list:
            return predictions
        else:
            return predictions[0]


if __name__ == '__main__':
    from classifier.train import load_examples

    examples = load_examples()

    predictor = Predictor()
    predictions = predictor(examples[10:])
    print(predictions)