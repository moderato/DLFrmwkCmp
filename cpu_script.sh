cnn="idsia"
size_x=48
size_y=48
dataset="GT"
epoch=25
batch=64
process=0 # 0 for none, 1 for 1-sigma, 2 for 2-sigma, 3 for clahe
print=0

python Neon.py $cnn $size_x $size_y $dataset $epoch $batch $process $print cpu
python PyTorch.py $cnn $size_x $size_y $dataset $epoch $batch $process $print cpu
python Mxnet.py $cnn $size_x $size_y $dataset $epoch $batch $process $print cpu

source deactivate
source activate cntk

export MKL_THREADING_LAYER=GNU

python Keras.py $cnn $size_x $size_y $dataset $epoch $batch $process $print theano tensorflow
python Cntk.py $cnn $size_x $size_y $dataset $epoch $batch $process $print