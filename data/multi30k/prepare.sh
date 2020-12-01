#!/bin/bash

# Export moses path
MOSES_PATH=../moses-5cbafabfd/scripts
PATH=${MOSES_PATH}:$PATH
SUFF="lc.norm.tok"

for tlang in de fr cs; do
  echo "Preparing en-${tlang} dataset"
  folder="en-${tlang}"
  mkdir -p $folder
  for sp in train val test_2016_flickr test_2017_flickr test_2017_mscoco test_2018_flickr; do
    # Process both sides
    for llang in en ${tlang}; do
      inp="raw/${sp}.${llang}.gz"
      if [ -f $inp ]; then
        zcat $inp | lowercase.perl -l ${llang} | normalize-punctuation.perl -l ${llang} | \
          tokenizer.perl -l ${llang} -a -threads 4 > $folder/${sp}.${SUFF}.${llang}
      fi
    done

    trg="${sp}.${SUFF}.${tlang}"

    # De-hyphenize test set targets for proper evaluation afterwards
    if [[ "$sp" =~ ^test.* ]] && [[ -f "${folder}/${trg}" ]]; then
      sed -r 's/\s*@-@\s*/-/g' < ${folder}/${trg} > ${folder}/${trg}.dehyph
    fi
  done
  # Create vocabularies
  pysimt-build-vocab ${folder}/train.${SUFF}.en -o ${folder}
  pysimt-build-vocab ${folder}/train.${SUFF}.${tlang} -o ${folder}
done

### Download features
pushd features
wget "https://zenodo.org/record/4298396/files/multi30k_butd_features.tar.bz2?download=1" -O butd.tar.bz2
tar xvf butd.tar.bz2
# rename folder
mv multi30k_butd_features butd
wget "https://zenodo.org/record/4298396/files/multi30k_resnet50_features.tar.bz2?download=1" -O resnet.tar.bz2
tar xvf resnet.tar.bz2
popd
