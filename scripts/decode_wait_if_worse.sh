#!/bin/bash

# Set GPU0 if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
test_set="test_2016_flickr,test_2017_flickr,test_2017_mscoco"

# Simultaneous greedy decode (Cho and Esipova, 2016)
# only decode SNMT models and not waitk
for ckpt in `find -L -name '*simultaneousnmt-*.best.loss.ckpt'`; do
  fname=`basename $ckpt`
  prefix=${ckpt/.best.loss.ckpt/}
  log=${ckpt/.best.loss.ckpt/.log}
  grep -q 'Training finished' ${log}
  if [ "$?" == "0" ]; then
    # check for the availabilty of one test set
    if [ ! -f "${prefix}.test_2017_flickr.s1_d1_wait_if_worse.gs" ]; then
      pysimt translate -m 60 -s ${test_set} -b 1 -f sgs --n-init-tokens "1,2" \
        --delta "1" --criteria "wait_if_worse" -o ${prefix} $ckpt
    fi
  fi
done
