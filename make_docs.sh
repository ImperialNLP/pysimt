#!/bin/bash

rm -rf docs/
pdoc --html pysimt -o docs/
mv docs/pysimt/* docs
