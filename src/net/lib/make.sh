TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_PATH=/usr/local/cuda/
CXXFLAGS=''

if [[ "$OSTYPE" =~ ^darwin ]]; then
        CXXFLAGS+='-undefined dynamic_lookup'
fi

cd roi_pooling_layer

if [ -d "$CUDA_PATH" ]; then
        nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
                -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
                -arch=sm_86

        g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc  -D_GLIBCXX_USE_CXX11_ABI=0\
                roi_pooling_op.cu.o -I $TF_INC -L $TF_LIB -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS \
                -lcudart -L $CUDA_PATH/lib64 -l:libtensorflow_framework.so.1
else
        g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
                -I $TF_INC -fPIC $CXXFLAGS
fi

cd ..
