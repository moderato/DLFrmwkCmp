cnn="idsia"
size_x=48
size_y=48
dataset="GT"
epoch=2
batch=64
process=0 # 0 for none, 1 for 1-sigma, 2 for 2-sigma, 3 for clahe
print=0

export OMP_NUM_THREADS=8
export KMP_AFFINITY=compact,1,0,granularity=fine # for neon

python Neon.py $cnn $size_x $size_y $dataset $epoch $batch $process $print cpu
python PyTorch.py $cnn $size_x $size_y $dataset $epoch $batch $process $print cpu
python Mxnet.py $cnn $size_x $size_y $dataset $epoch $batch $process $print cpu

source deactivate
source activate cntk

export MKL_THREADING_LAYER=GNU # for theano

python Keras.py $cnn $size_x $size_y $dataset $epoch $batch $process $print theano tensorflow
python Cntk.py $cnn $size_x $size_y $dataset $epoch $batch $process $print