#python3 main.py -data ml20m -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 500 -n_epoches 200
python3 main.py -data gowalla -model MultiVAE -annealing False -gpu 0 -train_batch_size 500 -n_val_samples 30
python3 main.py -data gowalla -model Multi_our_VAE -annealing False -K 3 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300
python3 main.py -data gowalla -model Multi_our_VAE -annealing False -K 3 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300

python3 main.py -data gowalla -model MultiVAE -annealing True -gpu 0 -train_batch_size 500 -n_val_samples 30 -anneal_cap 0.25
python3 main.py -data gowalla -model Multi_our_VAE -annealing True -K 3 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300
python3 main.py -data gowalla -model Multi_our_VAE -annealing True -K 3 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300