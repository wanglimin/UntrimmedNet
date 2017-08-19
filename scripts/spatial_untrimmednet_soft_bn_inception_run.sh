#!/usr/bin/env sh

TOOLS=/home/lmwang/code/yj_caffe/caffe_cvpr16/cmake_build_mem/install/bin

/usr/local/openmpi/bin/mpirun -n 8 \
$TOOLS/caffe train --solver=../models/spatial_untrimmednet_soft_bn_inception_solver.prototxt --weights=bn_inception_rgb_init.caffemodel
    
echo "Done."

