python ./video_sci_proxgrad.py \
--savepath ./save/test_ffdnet/ \
--testpath ../data/test_gray/ \
--loadpath ./models/ffdnet.ckpt \
--denoiser ffdnet \
--and_maxiters 180 \
--inference True
