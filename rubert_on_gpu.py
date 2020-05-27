import argparse
import numpy as np
import pandas as pd
import os
import codecs
from tqdm import tqdm
from keras_bert import Tokenizer
from tensorflow import keras
from keras_radam import RAdam
from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def get_args():
    parser = argparse.ArgumentParser(description="Pipeline for training Rubert on text corpus for classification task.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataframe_path", "-df", type=str, required=True,
                        help="path to the dataframe")
    parser.add_argument("--text_col", "-tcol", type=str, required=True,
                        help="name of column with texts")
    parser.add_argument("--label_col", "-lcol", type=str, required=True,
                        help="name of column with labels")
    parser.add_argument("--seq_len", "seq", type=int, default=128, required=False,
                        help="maximum sequence length")
    parser.add_argument("--batch_size", "-b", type=int, default=32, required=False,
                        help="batch_size")
    parser.add_argument("--epochs", "-e", type=int, default=30, required=False,
                        help="epochs num")
    parser.add_argument("--validation_split", "-split", type=float, default=0.2, required=False,
                        help="Part of data for validation")
    parser.add_argument("--checkpoints", "-cp", type=bool, default=True, required=False,
                        help="Checkpoint model after each epoch")
    parser.add_argument("--early_stopping", "-es", type=bool, default=True, required=False,
                        help="Stop training if loss stop decreasing")
    args = parser.parse_args()
    
    return args

def get_tokenizer_and_model(
			outputs_num, 
			loss='sparse_categorical_crossentropy', 
			metrics='sparse_categorical_accuracy'
			):

	pretrained_path = 'rubert_cased_L-12_H-768_A-12_v2'
	config_path = os.path.join(pretrained_path, 'bert_config.json')
	checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
	vocab_path = os.path.join(pretrained_path, 'vocab.txt')

	token_dict = {}
	with codecs.open(vocab_path, 'r', 'utf8') as reader:
	    for line in tqdm(reader):
	        token = line.strip()
	        token_dict[token] = len(token_dict)

	tokenizer = Tokenizer(token_dict)

    model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=True,
        trainable=True,
        seq_len=SEQ_LEN,
    )

	inputs  = model.inputs[:2]
    dense   = model.get_layer('NSP-Dense').output
    outputs = keras.layers.Dense(units=outputs_num, activation='relu')(dense)
    
    model = keras.models.Model(inputs, outputs)
    model.compile(
        RAdam(lr=LR),
        loss=loss,
        metrics=[metrics]
    )

    return model, tokenizer
    

def main():
	args = get_args()

	DF_PATH    = args.dataframe_path
	TEXT_COL   = args.text_col
	LABEL_COL  = args.label_col
	SEQ_LEN    = args.seq_len
	BATCH_SIZE = args.batch_size
	EPOCHS     = args.epochs
	VAL_SPLIT  = args.validation_split
	USE_EARLY_STOPPING = args.checkpoints
	USE_CHECKPOINTS    = args.early_stopping

	dataframe    = pd.read_csv(DF_PATH)
	input_texts  = df[TEXT_COL]
	input_labels = df[LABEL_COL]
	del(dataframe)

	indices, labels = [], []
	for i in tqdm(range(df.shape[0])):
	  ids, _ = tokenizer.encode(input_texts[i], max_len=SEQ_LEN)
	  indices.append(ids)
	  labels.append(input_labels[i])

	items = list(zip(indices, labels))
	np.random.seed(42)
	np.random.shuffle(items)
	indices, labels = zip(*items)
	indices = np.array(indices)
	mod = indices.shape[0] % BATCH_SIZE
	if mod > 0:
	    indices, labels = indices[:-mod], labels[:-mod]

	raw_split = indices.shape[0]*VAL_SPLIT
	batches_on_validation = int(raw_split // BATCH_SIZE)
	split = int(indices.shape[0] - BATCH_SIZE * batches_on_validation)

	X_train, y_train = [indices[:split], np.zeros_like(indices[:split])], np.array(labels)[:split]
	X_test, y_test   = [indices[split:], np.zeros_like(indices[split:])], np.array(labels)[split:]

	X, y = [indices, np.zeros_like(indices)], np.array(labels)

	bert_model, bert_tok = get_tokenizer_and_model(outputs_num=np.unique(labels))

	callbacks = []
	if USE_CHECKPOINTS:
		callbacks.append(ModelCheckpoint(
								"{epoch:03d}-{val_loss:.3f}.hdf5",
								monitor="val_loss",
								verbose=1,
								save_best_only=True,
								mode="auto")
						)

	if USE_EARLY_STOPPING:
		callbacks.append(EarlyStopping(
								patience=20, 
								monitor='val_loss',
								restore_best_weights=True)
						)

	history = bert_model.fit(
	    X_train,
	    y_train,
	    epochs=EPOCHS,
	    batch_size=BATCH_SIZE,
	    validation_data=(X_test, y_test)
	    callbacks=callbacks
	)

	with open('history.pickle', 'wb') as f:
		pickle.dump(history, f)

if __name__ = '__main__':
	main()


