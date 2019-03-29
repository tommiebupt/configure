#!/bin/bash

TENSORFLOW_COMMIT=9e76bf324f6bac63137a02bb6e6ec9120703ea9b # August 16, 2017
TENSORFLOW_SERVING_COMMIT=267d682bf43df1c8e87332d3712c411baf162fe9 # August 18, 2017
MODELS_COMMIT=78007443138108abf5170b296b4d703b49454487 # July 25, 2017

if [ -z $TENSORFLOW_SERVING_REPO_PATH ]; then
	TENSORFLOW_SERVING_REPO_PATH="serving"
fi
INITIAL_PATH=$(pwd)
export TF_NEED_CUDA=1
export TF_NCCL_VERSION=1.3
export NCCL_INSTALL_PATH=/usr/local/cuda-8.0
export TF_NEED_GCP=1
export TF_NEED_JEMALLOC=1
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_NEED_MKL=0
export TF_NEED_VERBS=0
export TF_NEED_MPI=0
export TF_NEED_GDR=0
export TF_ENABLE_XLA=0
export TF_CUDA_VERSION=8.0
export TF_CUDNN_VERSION=6
export TF_CUDA_CLANG=0
export TF_CUDA_COMPUTE_CAPABILITIES="6.1,6.1,6.1,6.1"
CUDA_PATH="/usr/local/cuda"
if [ ! -d $CUDA_PATH ]; then
	CUDA_PATH="/opt/cuda"
fi
export CUDA_TOOLKIT_PATH=$CUDA_PATH
export CUDNN_INSTALL_PATH=$CUDA_PATH
export GCC_HOST_COMPILER_PATH=$(which gcc)
export PYTHON_BIN_PATH=$(which python)
export CC_OPT_FLAGS="-march=native"


function python_path {
  "$PYTHON_BIN_PATH" - <<END
from __future__ import print_function
import site
import os
try:
  input = raw_input
except NameError:
  pass
python_paths = []
if os.getenv('PYTHONPATH') is not None:
  python_paths = os.getenv('PYTHONPATH').split(':')
try:
  library_paths = site.getsitepackages()
except AttributeError:
 from distutils.sysconfig import get_python_lib
 library_paths = [get_python_lib()]
all_paths = set(python_paths + library_paths)
paths = []
for path in all_paths:
  if os.path.isdir(path):
    paths.append(path)
if len(paths) == 1:
  print(paths[0])
else:
  ret_paths = ",".join(paths)
  print(ret_paths)
END
}

export PYTHON_LIB_PATH=$(python_path)


#cd $TENSORFLOW_SERVING_REPO_PATH
#cd tensorflow
#git reset --hard
#git fetch
#cd ../tf_models
#git reset --hard
#git fetch
##cd ..
#git reset --hard
#git fetch
#git submodule update --init --recursive
#git checkout $TENSORFLOW_SERVING_COMMIT
#git submodule update --init --recursive
#cd tf_models
#git checkout $MODELS_COMMIT
#cd ../tensorflow
#git submodule update --init --recursive
#git checkout $TENSORFLOW_COMMIT
#git submodule update --init --recursive
#wget -O /tmp/0002-TF-1.3-CUDA-9.0-and-cuDNN-7.0-support.patch https://github.com/tensorflow/tensorflow/files/1253794/0002-TF-1.3-CUDA-9.0-and-cuDNN-7.0-support.patch.txt
#git apply /tmp/0002-TF-1.3-CUDA-9.0-and-cuDNN-7.0-support.patch
#./configure
#cd ..

# force to use gcc-5 to compile CUDA
#if [ -e $(which gcc-5) ]; then
#	sed -i.bak 's/"gcc"/"gcc-5"/g' tensorflow/third_party/gpus/cuda_configure.bzl
#fi

## Error fix.  Ref: https://github.com/tensorflow/serving/issues/327
#sed -i.bak 's/external\/nccl_archive\///g' tensorflow/tensorflow/contrib/nccl/kernels/nccl_manager.h
#sed -i.bak 's/external\/nccl_archive\///g' tensorflow/tensorflow/contrib/nccl/kernels/nccl_ops.cc
## Error fix. Ref: https://github.com/tensorflow/tensorflow/issues/12979
#sed -i '\@https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz@d' tensorflow/tensorflow/workspace.bzl


#cd ..
#bazel --output_user_root=/export/software/tools/bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --jobs 6 --verbose_failures //tensorflow_serving/model_servers:tensorflow_model_server
# build fails, apply eigen patch
#wget -O /tmp/eigen.f3a22f35b044.cuda9.diff https://storage.googleapis.com/tf-performance/public/cuda9rc_patch/eigen.f3a22f35b044.cuda9.diff
#cd -P bazel-out/../../../external/eigen_archive
#patch -p1 < /tmp/eigen.f3a22f35b044.cuda9.diff
#cd -
#wget -O /tmp/eigen.f3a22f35b044.cuda9.diff https://storage.googleapis.com/tf-performance/public/cuda9rc_patch/eigen.f3a22f35b044.cuda9.diff
## build again
bazel --output_user_root=/export/software/tools/bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --jobs 6 --verbose_failures //tensorflow_serving/model_servers:tensorflow_model_server
#
#cd $INITIAL_PATH
