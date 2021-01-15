# GNN_sim


## Requirements

* Python 3.6+
* Tensorflow/Tensorflow-gpu 1.12.0
* Scipy 1.5.1

## Usage

Download pre-trained word embeddings `glove.6B.300d.txt` from [here](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to the repository.

Build graphs from the datasets in `data/corpus/` as:

    python build_graph.py [DATASET] [WINSIZE]

The datasets used in our report are 'question1_sub_bal' and 'question2_sub_bal'.


Start training and inference as:

    python train.py [--learning_rate LR]
                    [--epochs EPOCHS] [--batch_size BATCHSIZE]
                    [--hidden HIDDEN] [--steps STEPS]
                    [--dropout DROPOUT] [--weight_decay WD]

To reproduce our results, use 0.005 as LR, 50 epochs, a batch size of 512, a hidden size of 96 and a dropout of 0.25. No weight decay was used.
