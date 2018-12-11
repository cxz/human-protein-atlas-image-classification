#!/bin/bash

#
# resize all images to given SIZE
#
#
SIZE=224

#
# source directory
TRAIN_SRC=/kaggle1/train

#
# destination directory
TRAIN_DEST=/kaggle1/train-$SIZE

#
# filter source files by extension
EXT=png


#mkdir -p $DEST
#rm -f $DEST/*$EXT
#find $SRC -name "*$EXT" | parallel --no-notice convert {} -resize "\"$SIZE^>\"" $DEST/{/}


#
#
# same thing for test images
#
#
TEST_SRC=/opt/kaggle/human-protein-atlas-image-classification/input/test
TEST_DEST=/kaggle1/test-$SIZE

mkdir -p $TEST_DEST
rm -f $TEST_DEST/*$EXT
find $TEST_SRC -name "*$EXT" | parallel --no-notice convert {} -resize "\"$SIZE^>\"" $TEST_DEST/{/}

