mkdir -p output
python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-backprop --do-backprop-norm --do-info \
    ../max_out ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list /home/ubuntu/sdf/results conv4 \
    </dev/null >output/crop_max_patches.out 2>output/crop_max_patches.err &

python crop_max_patches.py --N 9 --gpu \
    --do-maxes --do-deconv --do-deconv-norm --do-backprop --do-backprop-norm --do-info \
    ../max_out ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list /home/ubuntu/sdf/results conv5
    </dev/null >output/crop_max_patches.out 2>output/crop_max_patches.err &

