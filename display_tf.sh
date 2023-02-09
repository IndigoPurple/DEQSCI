args_count=0
logdir=""

for var in "$@"
do
    let "args_count++"
    logdir="${logdir}$(basename ${var}):$var,"
done

if [ $args_count -eq 0 ]
then
    echo "please choose a checkpoint"
    exit 128
else
    logdir="${logdir::-1}"
    # echo "tensorboard --logdir_spec=$logdir --port=2001 --path_prefix=$tensorboard_base_url/web1"
    # tensorboard --logdir_spec=$logdir --port=2001 --path_prefix=$tensorboard_base_url/web1 --max_reload_threads 8
    tensorboard --logdir_spec=$logdir --port=2001 --max_reload_threads 8
fi
