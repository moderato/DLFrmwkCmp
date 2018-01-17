cnn="resnet-32"
size_xy=64
dataset="GT"
epoch=25
batch=64
process=3 # 0 for none, 1 for 1-sigma, 2 for 2-sigma, 3 for clahe
print=0

export OMP_NUM_THREADS=4
export KMP_AFFINITY=compact,1,0,granularity=fine # For neon

# export MKL_THREADING_LAYER=GNU # For theano

# THEANO_FLAGS='floatX=float32,device=cuda0' python Keras.py $cnn $size_xy $dataset $epoch $batch $process $print theano tensorflow

# python Cntk.py $cnn $size_xy $dataset $epoch $batch $process $print
python Keras.py $cnn $size_xy $dataset $epoch $batch $process $print tensorflow

python Neon.py $cnn $size_xy $dataset $epoch $batch $process $print mkl
# python PyTorch.py $cnn $size_xy $dataset $epoch $batch $process $print gpu
# python Mxnet.py $cnn $size_xy $dataset $epoch $batch $process $print gpu
