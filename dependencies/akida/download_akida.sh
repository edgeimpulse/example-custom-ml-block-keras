#!/bin/bash
set -e

cd ../

# Download the library
wget https://files.pythonhosted.org/packages/49/ee/af850e6d8e787aaef23d413d482dc11009784b35c8a33ed8b6bfb87c2ec6/cnn2snn-2.3.3.tar.gz
tar -xvzf cnn2snn-2.3.3.tar.gz
rm cnn2snn-2.3.3.tar.gz

ls cnn2snn-2.3.3
cat cnn2snn-2.3.3/setup.py

# Patch it to be compatible with TF2.11
rm cnn2snn-2.3.3/setup.py
cp akida/cnn2snn/setup.py cnn2snn-2.3.3
rm cnn2snn-2.3.3/cnn2snn.egg-info/requires.txt
cp akida/cnn2snn/requires.txt cnn2snn-2.3.3/cnn2snn.egg-info
