mkdir data\raw
wget https://data.deepai.org/mnist.zip --no-check-certificate -O data\raw\mnist.zip
tar -xzf data\raw\mnist.zip --directory data\raw
gzip -d data\raw\*.gz
del data\raw\mnist.zip
