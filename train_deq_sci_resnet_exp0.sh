python ./video_sci_proxgrad_multiGPU.py \
--batch_size 4 \
--lr 0.001 \
--lr_gamma 0.1 \
--sched_step 10 \
--savepath ./save/resnet_exp0/ \
--trainpath ../../data/DAVIS/matlab/ \
--testpath ../../data/test_gray/ \
--loadpath None \
--denoiser resnet