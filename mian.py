import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


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


def data_preprocessing(train_data: pd.DataFrame, test_data: pd.DataFrame, label_col: str, text_column: str):
	cleaned_train_data = [data_clean(w) for w in train_data[text_column]]
	cleaned_test_data = [data_clean(w) for w in test_data[text_column]]

	x_train, x_validaion, y_train, y_validaion = train_test_split(
		cleaned_train_data, train_data[label_col], test_size=0.1, random_state=255, shuffle=True)

	tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), norm='l2')

	tfidf_train = tfidf_vectorizer.fit_transform(x_train)

	tfidf_validation = tfidf_vectorizer.transform(x_validaion)

	tfidf_test = tfidf_vectorizer.transform(cleaned_test_data)

	return tfidf_train, tfidf_validation, y_train, y_validaion, tfidf_test


def main():
	train_file_path = "../data/train.csv"
	train_data_cols = ["id", "keyword", "location", "text"]
	train_label = "target"
	train_cols = [train_label] + train_data_cols
	test_file_path = "../data/test.csv"
	test_cols = train_data_cols

	train_df = read_data(train_file_path, train_cols)
	test_df = read_data(test_file_path, test_cols)

	preprocessing_result = data_preprocessing(train_df, test_df, 'target', 'text')
	tfidf_train, tfidf_validation, y_train, y_validaion, tfidf_test = preprocessing_result


	linear_svc = LinearSVC(random_state=55, loss='hinge')

	classifier = CalibratedClassifierCV(linear_svc, method='sigmoid', cv=2)

	classifier.fit(tfidf_train, y_train)

	pred_validation = classifier.predict(tfidf_validation)
	accuracy = metrics.accuracy_score(y_validaion, pred_validation)

	print("accuracy", accuracy)

	pred = classifier.predict(tfidf_test)


	with open('output.csv', 'w') as file:
		_str = ','.join(['id', 'target'])
		file.write(_str + '\n')

		for i in range(len(pred)):
			_str = ','.join([str(test_df['id'][i]), str(pred[i])])
			file.write(_str)
			file.write('\n')


if __name__ == '__main__':
	main()
