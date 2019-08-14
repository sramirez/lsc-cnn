vid_dir='/media/dh/DATA4TB/CrowdCounting/raw'

for f in $vid_dir/*.mp4; do
    echo Processing $f now..
    CUDA_VISIBLE_DEVICES=0 python3 crowdcount_video.py $f --cfg ndp.cfg
done

