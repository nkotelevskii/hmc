from tqdm import tqdm
from factory import get_models, get_optimizers
from aux import aux_epoch_cycle_enter, aux_batch_cycle_enter, aux_batch_cycle_exit, aux_epoch_cycle_exit
from states import get_initial_state, get_final_state
from optimization import run_optimization
from models import run_encoder, save_models

def run_training_vae(args):
    # define models
    models = get_models(args)  # returns dict, "encoder", "decoder" and so on...

    # define optimizers
    optimizers = get_optimizers(models, args)  # returns optimizers

    best_elbo = -float("inf")
    current_elbo_val = -float("inf")
    # training cycle
    for ep in tqdm(range(args.num_epoches)):
        aux_epoch_cycle_enter(ep, args)  # some actions
        for b_num, batch_train in enumerate(args.dataset):
            aux_batch_cycle_enter(ep, b_num, args)  # some action (like perform vanilla vae training), depending on epoch number
            enc_output = run_encoder(models['encoder'], args)  # it returns dict of arguments
            initial_state = get_initial_state(enc_output, args)
            final_state = get_final_state(initial_state, models)
            run_optimization(final_state, optimizers, args)
            aux_batch_cycle_exit(ep, b_num,
                                 args)  # some action (like perform vanilla vae training), depending on epoch number
        aux_epoch_cycle_exit(ep, current_elbo_val, best_elbo, args)  # some actions (like validation)
    save_models(models, args)
    print("\n Training is successfully finished!! \n")
