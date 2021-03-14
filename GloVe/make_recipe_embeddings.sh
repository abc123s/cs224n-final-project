#!/bin/bash
set -e

# global settings
VOCAB_MAX_SIZE=400000
VECTOR_SIZE=100
MAX_ITER=50
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=100
VERBOSE=2
MEMORY=4.0

# make programs
echo "Make programs"
make

BUILDDIR=build

# train glove vectors for every corpus
mkdir -p ./recipe_embeddings

for CORPUS_NAME in `ls corpuses`
do
  echo "Training embeddings for corpus $CORPUS_NAME"
  CORPUS=./corpuses/"$CORPUS_NAME"

  mkdir -p ./training_artifacts/"$CORPUS_NAME"
  VOCAB_FILE=./training_artifacts/"$CORPUS_NAME"/vocab.txt
  COOCCURRENCE_FILE=training_artifacts/"$CORPUS_NAME"/cooccurrence.bin
  COOCCURRENCE_SHUF_FILE=training_artifacts/"$CORPUS_NAME"/cooccurrence.shuf.bin

  SAVE_FILE=./recipe_embeddings/"${CORPUS_NAME}_${VECTOR_SIZE}d"

  echo
  echo "$ $BUILDDIR/vocab_count -max-vocab $VOCAB_MAX_SIZE -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
  $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
  echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
  $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
  echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
  $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
done
