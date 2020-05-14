import numpy as np
import pandas as pd
import re
import string
import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt


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


def plot_model_history(model_history):
	fig, axs = plt.subplots(1, 2, figsize=(15, 5))
	# summarize history for accuracy
	axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
	axs[0].set_title('Model Accuracy')
	axs[0].set_ylabel('Accuracy')
	axs[0].set_xlabel('Epoch')
	axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1), len(model_history.history['accuracy']) / 10)
	axs[0].legend(['train', 'val'], loc='best')
	# summarize history for loss
	axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
	axs[1].set_title('Model Loss')
	axs[1].set_ylabel('Loss')
	axs[1].set_xlabel('Epoch')
	axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
	axs[1].legend(['train', 'val'], loc='best')
	plt.show()


def build_model_CNN_EmbMatrix(max_features=10000,
							  embedding_size=512,
							  kernel_size1=2,
							  kernel_size2=4,
							  filters1=128,
							  filters2=64,
							  max_len=300,
							  embedded_matrix=None):
	print("creating model ...")

	model = Sequential()
	model.add(layers.Embedding(max_features,
							   embedding_size,
							   weights=[embedded_matrix],
							   input_length=max_len))
	model.add(layers.Conv1D(filters1,
							kernel_size1,
							activation='relu',
							padding='valid'))
	model.add(layers.MaxPooling1D())
	model.add(layers.Dropout(0.5))
	model.add(layers.Conv1D(filters2,
							kernel_size2,
							activation='relu',
							padding='valid'))
	model.add(layers.MaxPooling1D())
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.summary()

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


def processing_data_texts_to_sequences(df, max_len=300):
	print('Preprocessing data...')

	y = df['target'].values
	data_f = [data_clean(data) for data in df['text'].values]

	t = Tokenizer()
	t.fit_on_texts(data_f)

	encoded_data = t.texts_to_sequences(data_f)

	x_train, x_test, y_train, y_test = train_test_split(
		encoded_data,
		y,
		test_size=0.2,
		random_state=125,
		shuffle=True)

	x_train = sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = sequence.pad_sequences(x_test, maxlen=max_len)

	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)

	return x_train, x_test, y_train, y_test, t


def classifier_glove(dataset=None, max_len=300, epochs=3, batch_size=16):
	print('Preprocessing data...')

	x_train, x_test, y_train, y_test, t = processing_data_texts_to_sequences(dataset, max_len=max_len)

	embedded_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

	embedded_matrix = np.zeros((len(t.word_index) + 1, max_len))

	for word, i in t.word_index.items():
		embedding_vector = None
		if word in embedded_model.vocab:
			embedding_vector = embedded_model.get_vector(word)

		if embedding_vector is not None:
			embedded_matrix[i] = embedding_vector

	logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

	model = build_model_CNN_EmbMatrix(
		embedded_matrix=embedded_matrix,
		embedding_size=max_len,
		max_len=max_len,
		max_features=len(t.word_index) + 1,
	)

	history = model.fit(x_train,
						y_train,
						epochs=epochs,
						batch_size=batch_size,
						callbacks=[tensorboard_callback],
						validation_split=0.2
						)

	print('Evaluate model...')
	mse = model.evaluate(x_test, y_test)

	print("err = ", mse)

	plot_model_history(history)


def main():
	train_file_path = "./data/train.csv"
	train_data_cols = ["id", "keyword", "location", "text"]
	train_label = "target"
	train_cols = [train_label] + train_data_cols
	train_df = read_data(train_file_path, cols=train_cols)

	classifier_glove(train_df)


if __name__ == '__main__':
	main()
