# config names
config_names=('cifar10_my_vgg16' 'cifar10_my_vgg16_bn' 'cifar10_my_resnet20') # 
test=''

for config_name in ${config_names[@]}
# if config_name in one_pair_names, only use --pair 1_2; otehrwise, use --pair 1_2 2_3 1_3
do
    echo "Running $config_name"
    if [[ " ${one_pair_names[@]} " =~ " ${config_name} " ]]; then
        python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 1_2 ${test}
    else
        python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 1_2 ${test}
        python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 2_3 ${test}
        python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 1_3 ${test}
    fi
done