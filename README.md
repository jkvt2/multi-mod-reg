# multi-mod-reg
Loss functions for regressing against multi-modal ground truths.

Note: some of the experiments were written in tensorflow1.14, and some in tensorflow2.2, but I lumped them all into one file, so the code definitely cannot be run in one go. Apologies! If you want to run the 2.2 ones, then you'll have to comment out the 1.14 ones, and vice versa.

run_l2_experiment(): 1.14
run_mmr_experiment(): 1.14
run_mb_ind_experiment(): 1.14
run_mb_joint_experiment(): 1.14
run_gmm_clsfy_experiment(): 1.14
run_nf_experiment(): 2.2

[Blogpost describing this](https://jkvt2.github.io/deeplearning/2020/06/25/multi-mod-reg.html)
[Followup post on normalising flows](https://jkvt2.github.io/deeplearning/2020/07/06/norm-flows.html)
