#!/bin/bash

#
# resize all images to given SIZE
#
#
SIZE=1024

#
# source directory
TRAIN_SRC=/kaggle2/hpaic/train-full

#
# destination directory
TRAIN_DEST=/kaggle2/hpaic/train-$SIZE

#
# filter source files by extension
EXT=tif


mkdir -p $TRAIN_DEST && rm -f $TRAIN_DEST/*$EXT
find $TRAIN_SRC -name "*$EXT" | parallel --no-notice convert {} -resize "\"$SIZE^>\"" $TRAIN_DEST/{/}


#
#
# same thing for test images
#
#
TEST_SRC=/kaggle2/hpaic/test-full
TEST_DEST=/kaggle2/hpaic/test-$SIZE

mkdir -p $TEST_DEST && rm -f $TEST_DEST/*$EXT
find $TEST_SRC -name "*$EXT" | parallel --no-notice convert {} -resize "\"$SIZE^>\"" $TEST_DEST/{/}

find . -name "*_blue.tif" | sort | perl -pi -e 's/\.\///g' | perl -pi -e 's/_blue\.tif//g' | xargs -i convert "{}"_red.tif "{}"_green.tif "{}"_blue.tif -set colorspace RGB -combine "{}"_rgb.tif

