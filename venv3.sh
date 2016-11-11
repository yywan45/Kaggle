#!/bin/bash

pip3 install pandas
# installs pytz, six, python-dateutil, numpy, pandas

pip3 install scipy

pip3 install matplotlib
# installs cycler, pyparsing, matplotlib

pip3 install scikit-learn

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl
# installs binary for tensorflow for Mac OS X, CPU only, Python 2.7   

pip3 install --upgrade $TF_BINARY_URL
# installs funcsigs, pbr, mock, protobuf, tensorflow
