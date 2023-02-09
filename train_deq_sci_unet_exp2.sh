python ./video_sci_proxgrad_multiGPU.py \
--batch_size 12 \
--lr 0.001 \
--lr_gamma 0.5 \
--sched_step 2 \
--savepath ./save/exp2_bsz12/ \
--trainpath ../../data/DAVIS/matlab/ \
--testpath ../../data/test_gray/ \
--loadpath None \
--denoiser unet