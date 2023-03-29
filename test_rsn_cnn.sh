python ./video_sci_proxgrad.py \
--savepath ./save/test_rsn_cnn/ \
--testpath ../data/test_gray/ \
--loadpath ./models/rsn_cnn.ckpt \
--denoiser RealSN_SimpleCNN \
--inference True