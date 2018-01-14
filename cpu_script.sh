cnn="resnet-32"
size_xy=64
dataset="GT"
epoch=25
batch=64
process=3 # 0 for none, 1 for 1-sigma, 2 for 2-sigma, 3 for clahe
print=0

export OMP_NUM_THREADS=4
export KMP_AFFINITY=compact,1,0,granularity=fine # for neon

# python Neon.py $cnn $size_xy $dataset $epoch $batch $process $print cpu
# python PyTorch.py $cnn $size_xy $dataset $epoch $batch $process $print cpu
# python Mxnet.py $cnn $size_xy $dataset $epoch $batch $process $print cpu

python PyTorch.py $cnn $size_xy $dataset $epoch $batch $process $print cpu

python Mxnet.py $cnn $size_xy $dataset $epoch $batch $process $print cpu

source deactivate

source activate cntk

# # export MKL_THREADING_LAYER=GNU # for theano

python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print

python Keras.py $cnn $size_xy $dataset $epoch $batch $process $print tensorflow


# size_xy=48

# python Keras.py $cnn $size_xy $dataset $epoch $batch $process $print tensorflow

# python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print