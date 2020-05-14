import numpy as np
import pandas as pd
import re
import string
import bert
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow_hub as hub
from tqdm import tqdm
from transformers import BertTokenizer
from collections import namedtuple
from bert.tokenization.bert_tokenization import FullTokenizer
from tensorflow.keras import backend as K
from bert import BertModelLayer

MAX_SEQ_LENGTH = 128
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
keras_payer_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

# Initialize session
# tf.compat.v1.disable_eager_execution()
# sess = tf.compat.v1.Session()
#
# bert_module = hub.Module(bert_path)
# tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
# vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
# tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

bert_layer = hub.KerasLayer(keras_payer_path, trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)


class BertLayer(layers.Layer):
	def __init__(self, n_fine_tune_layers=10, **kwargs):
		self.n_fine_tune_layers = n_fine_tune_layers
		self.trainable = True
		self.output_size = 768
		self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
		super(BertLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.bert = hub.Module(
			self.bert_path,
			trainable=self.trainable,
			name="{}_module".format(self.name)
		)
		trainable_vars = self.bert.variables

		# Remove unused layers
		trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

		# Select how many layers to fine tune
		trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

		# Add to trainable weights
		for var in trainable_vars:
			self._trainable_weights.append(var)

		# Add non-trainable weights
		for var in self.bert.variables:
			if var not in self._trainable_weights:
				self._non_trainable_weights.append(var)

		super(BertLayer, self).build(input_shape)

	def call(self, inputs):
		inputs = [K.cast(x, dtype="int32") for x in inputs]
		input_ids, input_mask, segment_ids = inputs
		bert_inputs = dict(
			input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
		)
		result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
			"pooled_output"
		]
		return result

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_size)


def read_data(file_path: str, cols: [str]) -> pd.DataFrame:
	df = pd.read_csv(file_path, usecols=cols)  # nrows=3000
	return df


def data_clean(text: str):
	text = re.sub('<[^<]+?>', ' ', text)

	text = text.replace('\\"', '')

	text = text.replace('\n', ' ')

	text = text.replace('\t', ' ')

	text = text.replace('"', '')

	text = text.translate(str.maketrans('', '', string.punctuation))

	text = re.sub(' +', ' ', text)

	text = re.sub('\d+', '0', text)

	text = text.lower()

	return text


def cerate_model():
	input_word_ids = Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_word_ids")
	input_mask = Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="input_mask")
	segment_ids = Input(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32, name="segment_ids")

	bert_input, bert_output = bert_layer([input_word_ids, input_mask, segment_ids])
	print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++", bert_output.shape)
	x = layers.Dense(256, activation="relu")(bert_output)
	x = layers.Dropout(0.2)(x)
	out = layers.Dense(1, activation="sigmoid", name="dense_output")(x)

	model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
	model.summary()
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model



# bert_inputs = [input_word_ids, input_mask, segment_ids]
# bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
# dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
# pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)
#
# model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.summary()
#
# return model


def get_masks(tokens, max_seq_length):
	return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
	"""Segments: 0 for the first sequence, 1 for the second"""
	segments = []
	current_segment_id = 0
	for token in tokens:
		segments.append(current_segment_id)
		if token == "[SEP]":
			current_segment_id = 1
	return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
	"""Token ids from Tokenizer vocab"""
	token_ids = tokenizer.convert_tokens_to_ids(tokens, )
	input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
	return input_ids


def create_single_input(sentence, MAX_LEN):
	stokens = tokenizer.tokenize(sentence)

	stokens = stokens[:MAX_LEN]

	stokens = ["[CLS]"] + stokens + ["[SEP]"]

	ids = get_ids(stokens, tokenizer, MAX_SEQ_LENGTH)
	masks = get_masks(stokens, MAX_SEQ_LENGTH)
	segments = get_segments(stokens, MAX_SEQ_LENGTH)

	return ids, masks, segments


def create_input_array(sentences):
	input_ids, input_masks, input_segments = [], [], []

	for sentence in tqdm(sentences, position=0, leave=True):
		ids, masks, segments = create_single_input(sentence, MAX_SEQ_LENGTH - 2)

		input_ids.append(ids)
		input_masks.append(masks)
		input_segments.append(segments)

	return [np.asarray(input_ids, dtype=np.int32),
			np.asarray(input_masks, dtype=np.int32),
			np.asarray(input_segments, dtype=np.int32)]


def main():
	train_file_path = "./data/train.csv"
	train_data_cols = ["id", "keyword", "location", "text"]
	train_label = "target"
	train_cols = [train_label] + train_data_cols
	test_file_path = "./data/test.csv"
	test_cols = train_data_cols
	train_df = read_data(train_file_path, cols=train_cols)
	test_df = read_data(test_file_path, test_cols)
	test_sentences = [data_clean(w) for w in test_df['text'].values]

	train_sentences = [data_clean(w) for w in train_df['text'].values]
	train_y = train_df[train_label].values
	inputs = create_input_array(train_sentences)

	model = cerate_model()

	model.fit(inputs, train_y, epochs=2, batch_size=16)


if __name__ == '__main__':
	main()
