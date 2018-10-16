#!/usr/bin/env sh


wget -O models/anet1.2_spatial_untrimmednet_hard_bn_inception_train_only.caffemodel http://mcg.nju.edu.cn/models/UntrimmedNet/anet1.2_train_rgb_weak_tsn_bn_inception_average_seg3_top3_sw7_iter_10000.caffemodel
wget -O models/anet1.2_spatial_untrimmednet_soft_bn_inception_train_only.caffemodel http://mcg.nju.edu.cn/models/UntrimmedNet/anet1.2_train_rgb_weak_tsn_bn_inception_average_seg3_attention_sw7_iter_10000.caffemodel
wget -O models/anet1.2_temporal_untrimmednet_hard_bn_inception_train_only.caffemodel http://mcg.nju.edu.cn/models/UntrimmedNet/anet1.2_train_flow_weak_tsn_bn_inception_average_seg3_top3_sw7_iter_18000.caffemodel
wget -O models/anet1.2_temporal_untrimmednet_soft_bn_inception_train_only.caffemodel http://mcg.nju.edu.cn/models/UntrimmedNet/anet1.2_train_flow_weak_tsn_bn_inception_average_seg3_attention_sw7_iter_18000.caffemodel





