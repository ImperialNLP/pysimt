#!/bin/bash

# Set GPU0 if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
test_set="test_2016_flickr,test_2017_flickr,test_2017_mscoco"


# Train-time wait-k
for ckpt in `find -L -name '*simultaneouswaitk*.best.loss.ckpt'`; do
  fname=`basename $ckpt`
  model=`dirname $ckpt`
  k=`echo $model | sed -r 's#\./wait([0-9])-rnn.*#\1#'`
  prefix=${ckpt/.best.loss.ckpt/}
  log=${ckpt/.best.loss.ckpt/.log}
  grep -q 'Training finished' ${log}
  if [ "$?" == "0" ]; then
    # check for the availability of one test set
    if [ ! -f "${prefix}.test_2017_flickr.wait${k}.gs" ]; then
      pysimt translate -m 60 -s ${test_set} -b 1 -f wk --n-init-tokens "$k" \
        -o ${prefix} $ckpt
    fi
  fi
done
