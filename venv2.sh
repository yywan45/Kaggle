#!/bin/bash

pip install pandas
# installs pytz, six, python-dateutil, numpy, pandas

pip install scipy

pip install matplotlib
# installs cycler, pyparsing, matplotlib

pip install scikit-learn

export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py2-none-any.whl
# installs binary for tensorflow for Mac OS X, CPU only, Python 2.7

pip install --upgrade $TF_BINARY_URL
# installs funcsigs, pbr, mock, protobuf, tensorflow
