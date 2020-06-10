#python3 main.py -data gowalla -model Multi_our_VAE -annealing False -K 10 -N 1 -learnable_reverse False -gpu 0 -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 100 -n_epoches 300 -learnscale True
#python3 main.py -data gowalla -model Multi_our_VAE -annealing False -K 5 -N 1 -learnable_reverse False -gpu 0 -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 100 -n_epoches 300 -learnscale True
#
#python3 main.py -data gowalla -model Multi_our_VAE -annealing True -K 10 -N 1 -learnable_reverse False -gpu 0 -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 100 -n_epoches 300 -learnscale True
#python3 main.py -data gowalla -model Multi_our_VAE -annealing True -K 5 -N 5 -learnable_reverse False -gpu 0 -lrdec 3e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 100 -n_epoches 300 -learnscale True

python3 main.py -data ml20m -model Multi_our_VAE -K 10 -N 1 -gpu 0 -lrdec 3e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing True -train_batch_size 500 -n_epoches 300 -learnscale True