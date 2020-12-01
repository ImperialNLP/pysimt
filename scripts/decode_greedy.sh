#!/bin/bash

# Only decode for snmt models and not waitk. It does not make sense for the latter

# Set GPU0 if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
test_set="test_2016_flickr,test_2017_flickr,test_2017_mscoco"


# Greedy decode everything (batched)
for ckpt in `find -L -name '*simultaneousnmt-*.best.loss.ckpt'`; do
  fname=`basename $ckpt`
  prefix=${ckpt/.best.loss.ckpt/}
  log=${ckpt/.best.loss.ckpt/.log}
  grep -q 'Training finished' ${log}
  if [ "$?" == "0" ]; then
    # check for the availability of one test set
    if [ ! -f "${prefix}.test_2017_flickr.gs" ]; then
      pysimt stranslate -m 60 -s ${test_set} -f gs -o ${prefix} $ckpt
    fi
  fi
done
