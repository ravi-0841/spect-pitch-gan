import os
import numpy as np
import argparse
import sys
import scipy.io as scio

from nn_models.model_energy_f0_momenta_wasserstein import VariationalCycleGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


lp_dict = {
        'neu-ang': '1e-05', 
        'neu-hap': '0.0001',
        'neu-sad': '0.0001'
        }

le_dict = {
        'neu-ang': '0.1', 
        'neu-hap': '0.001',
        'neu-sad': '0.1'
        }

random_seed_dir = {
        'neu-ang': 21, 
        'neu-hap': 4, 
        'neu-sad': 11
        }

def evaluate(data_dir, fold=1, emo_pair='neu-ang', run=1):
    
    random_seed = random_seed_dir[emo_pair]

    data_valid = scio.loadmat(os.path.join(data_dir, emo_pair+'_fold_{}.mat'.format(fold)))

    pitch_A_valid = np.transpose(np.vstack(data_valid['valid_f0_feat_src']), axes=(0,2,1))
    pitch_B_valid = np.transpose(np.vstack(data_valid['valid_f0_feat_tar']), axes=(0,2,1))
    energy_A_valid = np.transpose(np.vstack(data_valid['valid_ec_feat_src'] + -1e-06), axes=(0,2,1))
    energy_B_valid = np.transpose(np.vstack(data_valid['valid_ec_feat_tar'] + -1e-06), axes=(0,2,1))
    mfc_A_valid = np.transpose(np.vstack(data_valid['valid_mfc_feat_src']), axes=(0,2,1))
    mfc_B_valid = np.transpose(np.vstack(data_valid['valid_mfc_feat_tar']), axes=(0,2,1))

    main_model_dir = './model/{}/sum_mfc_wstn_{}_fold_{}'.format(emo_pair, emo_pair, fold)
    sub_model_dir = 'lp_{}_le_{}_li_0.0_neu-ang_fold_{}_run_{}_random_seed_{}'.format(lp_dict[emo_pair], 
                                                                                    le_dict[emo_pair], fold, 
                                                                                    run, random_seed)

    for epoch in [100, 200, 300, 400]:
        print("emo_pair - {}".format(emo_pair))
        print("random_seed - {}".format(random_seed))
        print("run - {}".format(run))
        print("epoch - {}".format(epoch))
        
        #use pre_train arg to provide trained model
        model = VariationalCycleGAN(dim_pitch=1, dim_energy=1, 
                                    dim_mfc=23, mode='test')
        model_name = '{}_{}.ckpt'.format(emo_pair, epoch)
        model.load(filepath=os.path.join(main_model_dir, 
                                         sub_model_dir, 
                                         model_name))
        
        loss_pitch = []
        loss_energy = []
        
        for i in range(energy_A_valid.shape[0]):
            gen_pitch_A, gen_energy_A, \
            gen_pitch_B, gen_energy_B, \
            mom_pitch_A, mom_pitch_B, \
            mom_energy_A, mom_energy_B = model.test_gen(mfc_A=mfc_A_valid[i:i+1], 
                            mfc_B=mfc_B_valid[i:i+1], energy_A=energy_A_valid[i:i+1], 
                            energy_B=energy_B_valid[i:i+1], pitch_A=pitch_A_valid[i:i+1], 
                            pitch_B=pitch_B_valid[i:i+1])
            
            loss_pitch.append(np.sum((gen_pitch_B.reshape(-1,) - pitch_B_valid[i:i+1].reshape(-1,))**2))
            loss_energy.append(np.sum((gen_energy_B.reshape(-1,) - energy_B_valid[i:i+1].reshape(-1,))**2))
            
        print(np.mean(loss_pitch), np.std(loss_pitch))
        print(np.mean(loss_energy), np.std(loss_energy))
        sys.stdout.flush()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Evaluate VariationalCycleGAN model for datasets.')

    emo_dict = {
            "neu-ang":['neutral', 'angry'], 
            "neu-sad":['neutral', 'sad'], 
            "neu-hap":['neutral', 'happy']
            }

    emo_pair_default = "neu-ang"

    parser.add_argument('--emotion_pair', type=str, help="Emotion Pair", 
            default=emo_pair_default)
    parser.add_argument('--run', type=int, help='run', default=1)
    parser.add_argument('--fold', type=int, help='fold', default=1)
    
    argv = parser.parse_args()

    emo_pair = argv.emotion_pair
    train_dir = "./data/"+emo_pair

    run = argv.run
    fold = argv.fold

    evaluate(train_dir=train_dir, fold=fold, emo_pair=argv.emotion_pair, run=run)
