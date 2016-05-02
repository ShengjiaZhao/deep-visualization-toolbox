mkdir -p output
python analyze/find_activation.py --gpu --num 100 \
    ../models/caffenet-yos/caffenet-yos-deploy.prototxt ../models/caffenet-yos/caffenet-yos-weights \
    /home/ubuntu/sdf/images /home/ubuntu/sdf/database_list activation_out conv3,conv4,conv5 \
    </dev/null >output/activation.out 2>output/activation.err &