hostname

model='cifar_resnet20_ln'
dataset=cifar100
seed=20

cd	../
python	run_ex.py	\
--project	vf	\
--run-name	"cifar100_resnet20_ln_seed_${seed}"	\
--model	${model}	\
--dataset	${dataset}	\
--data-dir	../data	\
--epochs	160	\
--batch-size	128	\
--print-freq	100	\
--test-freq	10	\
--optimizer	sgd	\
--lr	0.1	\
--scheduler linear_step \
--warmup-iters 391 \
--milestones "32000, 48000" \
--momentum	0.9	\
--wd	"0.0001"	\
--seed	${seed}	\
--wandb-mode	online	\
--save-model	\
--save-freq	160	\
--save-dir ex_results/${dataset}/${model}/diff_init/seed_${seed} \
--diff-init \
--device "cuda:0" \
--train-only \
# --subset
