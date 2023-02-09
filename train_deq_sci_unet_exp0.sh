python ./video_sci_proxgrad_yaping.py \
--batch_size 64 \
--lr 0.001 \
--lr_gamma 0.9 \
--sched_step 10 \
--savepath ./save/test/ \
--trainpath ../../data/DAVIS/matlab/ \
--testpath ../../data/test_gray/ \
--loadpath None
