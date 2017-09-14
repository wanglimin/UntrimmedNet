#!/usr/bin/env sh

TOOLS=/home/lmwang/code/caffe_cvpr16/cmake_build_mem/install/bin

/usr/local/bin/mpirun -n 8 \
$TOOLS/caffe train --solver=../models/temporal_untrimmednet_soft_bn_inception_solver.prototxt --weights=bn_inception_flow_init.caffemodel
    
echo "Done."

