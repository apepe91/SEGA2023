#!/usr/bin/env bash

./build.sh

docker save segaalgorithm | gzip -c > SegaAlgorithm.tar.gz
