mkdir -p data/Decaf
wget -P data/Decaf https://vcai.mpi-inf.mpg.de/projects/Decaf/static/DecafDataset.zip
unzip data/Decaf/DecafDataset.zip -d data/Decaf
mv data/Decaf/DecafDataset data/Decaf/data
rm -f data/Decaf/DecafDataset.zip