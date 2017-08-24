# UntrimmedNet for Action Recognition and Detection
We provide the code and models for our CVPR paper ([Arxiv Preprint](https://arxiv.org/abs/1703.03329)):

      UntrimmedNets for Weakly Supervised Action Recognition and Detection
      Limin Wang, Yuanjun Xiong, Dahua Lin, and Luc Van Gool
      in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017
### Updates
- August 20th, 2017
  * Release the model protos.
  
### Guide
The training of UntrimmedNet is composed of three steps:
- **Step 1**: extract action proposals (or shot boundaries) for each untrimmed video. We provide a sample of detected shot boudary on the ActivityNet (v1.2) under the folders of `data/anet1.2/anet_1.2_train_window_shot/` and `data/anet1.2/anet1.2/anet_1.2_val_window_shot/`.
- **Step 2**: construct file lists for training and validation. There are two filelists: one containing file path, number of frames, and label; the other one containing the shot file path and number of frames (Examples are in the folder `data/anet1.2/`).
- **Step 3**: train UntrimmedNets using our modified caffe: https://github.com/yjxiong/caffe/tree/untrimmednet

The testing of UntrimmedNet for action recognition is based on temporal sliding window and top-k pooling

The testing of UntrimmedNet for action detection is based on  a simple baseline (see code in `matlab/`）

