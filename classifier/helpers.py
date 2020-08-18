from transformers.data.processors.utils import InputExample, InputFeatures
import tensorflow as tf

def features_to_dataset(features):
    def gen():
        for ex in features:
            yield  ({'input_ids': ex.input_ids,
                        'attention_mask': ex.attention_mask,
                        'token_type_ids': ex.token_type_ids},
                    ex.label)
    return tf.data.Dataset.from_generator(gen,
        ({'input_ids': tf.int32,
            'attention_mask': tf.int32,
            'token_type_ids': tf.int32},
            tf.int64),
        ({'input_ids': tf.TensorShape([None]),
            'attention_mask': tf.TensorShape([None]),
            'token_type_ids': tf.TensorShape([None])},
            tf.TensorShape([])))

def create_input_feature(tokenizer, output_mode, example, max_length, mask_padding_with_zero, pad_on_left, pad_token, pad_token_segment_id, label_map):
    example = InputExample(
        example['id'],
        example['sentence1'],
        example['sentence2'] if 'sentence2' in example else None,
        example['label']
    )

    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        truncation_strategy='only_first'  # We're truncating the first sequence in priority
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    if output_mode == "classification":
        label = label_map[example.label]
    elif output_mode == "regression":
        label = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label=label)

def convert_examples_to_features(tokenizer, output_mode, examples,
                                      max_length=512,
                                      label_list=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):

    if output_mode == 'classification':
        assert label_list is not None
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        assert label_list is None
        label_map = None

    features, labels = [], []
    for example in examples:
        input_feature = create_input_feature(tokenizer, output_mode, example, max_length, mask_padding_with_zero, pad_on_left, pad_token, pad_token_segment_id, label_map)
        features.append(input_feature)
        labels.append(example['label'])

    return features_to_dataset(features), labels