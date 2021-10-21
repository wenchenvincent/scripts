echo "Clear page cache"
sudo sync && sudo /sbin/sysctl vm.drop_caches=3

CURRENTDATE=`date +"%Y-%m-%d-%T"`
SECONDS=0

#Mention the number of GPU's you want to run on
num_of_gpus="8"

#Mention the batch_size
batch_size_per_gpu="256"

#Global batch_size
global_batch_size=$(echo $(( num_of_gpus * batch_size_per_gpu )))

python3 resnet_ctl_imagenet_main.py --base_learning_rate=9.1 --batch_size=$global_batch_size --clean --data_dir=/data/imagenet_tf/tf_records/ --datasets_num_private_threads=32 --dtype=fp16 --enable_eager --epochs_between_evals=1 --eval_dataset_cache --eval_offset_epochs=2 --eval_prefetch_batchs=192 --label_smoothing=0.1 --log_steps=125 --lr_schedule=polynomial --model_dir=/dockerx/benchmark_8_gpu --num_gpus=${num_of_gpus} --optimizer=LARS --report_accuracy_metrics=true --single_l2_loss_op --steps_per_loop=626 --data_format=channels_first --tf_gpu_thread_mode=gpu_private --train_epochs=39 --enable_device_warmup --training_dataset_cache --training_prefetch_batchs=128 --verbosity=0 --warmup_epochs=2 --weight_decay=0.0002 2>&1 | tee run.log.${CURRENTDATE}

