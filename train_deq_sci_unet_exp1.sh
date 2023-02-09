python ./video_sci_proxgrad_multiGPU.py \
--batch_size 4 \
--lr 0.0001 \
--lr_gamma 0.9 \
--sched_step 10 \
--savepath ./save/exp1_bsz4/ \
--trainpath ../../data/DAVIS/matlab/ \
--testpath ../../data/test_gray/ \
--loadpath None
