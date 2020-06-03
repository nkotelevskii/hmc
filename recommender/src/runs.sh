#python3 main.py -data ml20m -model Multi_our_VAE -K 2 -N 1 -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing False -train_batch_size 500 -n_epoches 300
#python3 main.py -data ml20m -model Multi_our_VAE -K 1 -N 2 -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing False -train_batch_size 500 -n_epoches 300
#python3 main.py -data ml20m -model Multi_our_VAE -K 1 -N 3 -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing False -train_batch_size 500 -n_epoches 300
#python3 main.py -data ml20m -model Multi_our_VAE -K 2 -N 2 -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing False -train_batch_size 500 -n_epoches 300

#python3 main.py -data foursquare -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 300
#python3 main.py -data gowalla -model MultiVAE -annealing False -n_val_samples 30 -gpu 0 -train_batch_size 300
#python3 main.py -data foursquare -model Multi_our_VAE -annealing False -K 2 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300

#python3 main.py -data foursquare -model Multi_our_VAE -annealing False -K 3 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300
#python3 main.py -data gowalla -model Multi_our_VAE -annealing False -K 4 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300
#python3 main.py -data gowalla -model Multi_our_VAE -annealing False -K 5 -N 1 -learnable_reverse False -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300

#python3 main.py -data ml20m -model Multi_our_VAE -K 2 -N 1 -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing True -train_batch_size 500
#python3 main.py -data ml20m -model Multi_our_VAE -K 3 -N 1 -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing True -train_batch_size 500 -anneal_cap 5.0
#python3 main.py -data ml20m -model Multi_our_VAE -K 2 -N 2 -gpu 0 -lrdec 1e-3 -lrenc 1e-3 -learnable_reverse False -n_val_samples 30 -learntransitions False -annealing True -train_batch_size 500 -anneal_cap 5.0

#python3 main.py -data foursquare -model MultiVAE -annealing True -gpu 1 -train_batch_size 300 -n_val_samples 30 -n_epoches 300
python3 main.py -data foursquare -model Multi_our_VAE -annealing True -K 2 -N 1 -learnable_reverse False -gpu 1 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300 -n_epoches 300 -learnscale True
python3 main.py -data foursquare -model Multi_our_VAE -annealing True -K 3 -N 1 -learnable_reverse False -gpu 1 -lrdec 1e-3 -lrenc 1e-3 -n_val_samples 30 -learntransitions False -train_batch_size 300 -n_epoches 300 -learnscale True