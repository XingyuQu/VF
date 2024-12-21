# config names
config_names=('cifar10_my_vgg16') # 
test=''


# for config_name in ${config_names[@]}
# # if config_name in one_pair_names, only use --pair 1_2; otehrwise, use --pair 1_2 2_3 1_3
# do
#     echo "Running $config_name"
#     if [[ " ${one_pair_names[@]} " =~ " ${config_name} " ]]; then
#         python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 1_2 ${test}
#     else
#         python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 1_2 ${test}
#         python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 2_3 ${test}
#         python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca/ --pair 1_3 ${test}
#     fi
# done

# one ref stats
for config_name in ${config_names[@]}
# if config_name in one_pair_names, only use --pair 1_2; otehrwise, use --pair 1_2 2_3 1_3
do
    echo "Running $config_name"
    if [[ " ${one_pair_names[@]} " =~ " ${config_name} " ]]; then
        python cifar_cca_experiments.py --device cuda:0 --config $config_name --save_dir pfm_results/cifar/cca_one_ref/ --pair 1_2 ${test} --one_ref_stats
    else
        python cifar_cca_experiments.py --device cuda:1 --config $config_name --save_dir pfm_results/cifar/cca_one_ref/ --pair 1_2 ${test} --one_ref_stats
        python cifar_cca_experiments.py --device cuda:1 --config $config_name --save_dir pfm_results/cifar/cca_one_ref/ --pair 2_3 ${test} --one_ref_stats
        python cifar_cca_experiments.py --device cuda:1 --config $config_name --save_dir pfm_results/cifar/cca_one_ref/ --pair 1_3 ${test} --one_ref_stats
    fi
done