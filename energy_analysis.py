import scipy.io.wavfile as scwav
import numpy as np
import pylab
import librosa
import pyworld as pw
import os
import scipy.io as scio

from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from sklearn.manifold import TSNE


def _power_to_db(S):
    return 20*np.log10(S)


def _get_spect(filename, dim=8, mfcc=True):
    sr, data = scwav.read(filename=filename)
    data = np.asarray(data, np.float64)
    _, spect, _ = pw.wav2world(data, sr, frame_period=5)
    if spect.shape[0] > 128:
        q = np.random.randint(0, spect.shape[0] - 128)
        spect = spect[q:q+128]
        u_mat, s_mat, v_mat = np.linalg.svd(spect)
        rank1_appx = s_mat[0] * np.dot(u_mat[:,0:1], v_mat[0:1,:])
        rank2_appx = rank1_appx + (s_mat[1] * np.dot(u_mat[:,1:2], v_mat[1:2,:]))
        rank3_appx = rank2_appx + (s_mat[2] * np.dot(u_mat[:,2:3], v_mat[2:3,:]))
        rank4_appx = rank3_appx + (s_mat[3] * np.dot(u_mat[:,3:4], v_mat[3:4,:]))
        rank5_appx = rank4_appx + (s_mat[4] * np.dot(u_mat[:,4:5], v_mat[4:5,:]))
        rank6_appx = rank5_appx + (s_mat[5] * np.dot(u_mat[:,5:6], v_mat[5:6,:]))
        rank7_appx = rank6_appx + (s_mat[6] * np.dot(u_mat[:,6:7], v_mat[6:7,:]))
        rank8_appx = rank7_appx + (s_mat[7] * np.dot(u_mat[:,7:8], v_mat[7:8,:]))
        
        if mfcc:
            mfc1 = pw.code_spectral_envelope(np.abs(rank1_appx), sr, dim)
            mfc2 = pw.code_spectral_envelope(np.abs(rank2_appx), sr, dim)
            mfc3 = pw.code_spectral_envelope(np.abs(rank3_appx), sr, dim)
            mfc4 = pw.code_spectral_envelope(np.abs(rank4_appx), sr, dim)
            mfc5 = pw.code_spectral_envelope(np.abs(rank5_appx), sr, dim)
            mfc6 = pw.code_spectral_envelope(np.abs(rank6_appx), sr, dim)
            mfc7 = pw.code_spectral_envelope(np.abs(rank7_appx), sr, dim)
            mfc8 = pw.code_spectral_envelope(np.abs(rank8_appx), sr, dim)
        else:
            mfc1 = rank1_appx
            mfc2 = None
            mfc3 = None
            mfc4 = None
            mfc5 = None
            mfc6 = None
            mfc7 = None
            mfc8 = None
            
        
        return [mfc1, mfc2, mfc3, mfc4, mfc5, mfc6, mfc7, mfc8]
    else:
        return None


def _get_spect_no_abs(filename, dim=8, mfcc=True):
    sr, data = scwav.read(filename=filename)
    data = np.asarray(data, np.float64)
    _, spect, _ = pw.wav2world(data, sr, frame_period=5)
    if spect.shape[0] > 128:
        q = np.random.randint(0, spect.shape[0] - 128)
        spect = spect[q:q+128]
        u_mat, s_mat, v_mat = np.linalg.svd(spect)
        rank1_appx = s_mat[0] * np.dot(u_mat[:,0:1], v_mat[0:1,:])
        rank2_appx = rank1_appx + (s_mat[1] * np.dot(u_mat[:,1:2], v_mat[1:2,:]))
        rank3_appx = rank2_appx + (s_mat[2] * np.dot(u_mat[:,2:3], v_mat[2:3,:]))
        rank4_appx = rank3_appx + (s_mat[3] * np.dot(u_mat[:,3:4], v_mat[3:4,:]))
        rank5_appx = rank4_appx + (s_mat[4] * np.dot(u_mat[:,4:5], v_mat[4:5,:]))
        rank6_appx = rank5_appx + (s_mat[5] * np.dot(u_mat[:,5:6], v_mat[5:6,:]))
        rank7_appx = rank6_appx + (s_mat[6] * np.dot(u_mat[:,6:7], v_mat[6:7,:]))
        rank8_appx = rank7_appx + (s_mat[7] * np.dot(u_mat[:,7:8], v_mat[7:8,:]))
        
        if mfcc:
            mfc1 = pw.code_spectral_envelope(rank1_appx, sr, dim)
            mfc2 = pw.code_spectral_envelope(rank2_appx, sr, dim)
            mfc3 = pw.code_spectral_envelope(rank3_appx, sr, dim)
            mfc4 = pw.code_spectral_envelope(rank4_appx, sr, dim)
            mfc5 = pw.code_spectral_envelope(rank5_appx, sr, dim)
            mfc6 = pw.code_spectral_envelope(rank6_appx, sr, dim)
            mfc7 = pw.code_spectral_envelope(rank7_appx, sr, dim)
            mfc8 = pw.code_spectral_envelope(rank8_appx, sr, dim)
        else:
            mfc1 = rank1_appx
            mfc2 = None
            mfc3 = None
            mfc4 = None
            mfc5 = None
            mfc6 = None
            mfc7 = None
            mfc8 = None
            
        
        return [mfc1, mfc2, mfc3, mfc4, mfc5, mfc6, mfc7, mfc8]

    else:
        return None


if __name__ == '__main__':
    
#    sample_rate = 16000
#    window_len = 0.005
    
#    wav_file = '38.wav'
#    files = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/neutral-angry/train/neutral', '*.wav')))
#    wav_files = [os.path.basename(f) for f in files]
#    
#    min_val = []
#    max_val = []
    
#    for w in wav_files:
#        src = scwav.read(os.path.join('/home/ravi/Downloads/Emo-Conv/neutral-angry/train/neutral', w))
#        src = np.asarray(src[1], np.float64)
#        f0_src, sp_src, ap_src = pw.wav2world(src, 16000, frame_period=5)
#        mfc_src = pw.code_spectral_envelope(sp_src, 16000, 23)
#        
#        tar = scwav.read(os.path.join('/home/ravi/Downloads/Emo-Conv/neutral-angry/train/angry', w))
#        tar = np.asarray(tar[1], np.float64)
#        f0_tar, sp_tar, ap_tar = pw.wav2world(tar, 16000, frame_period=5)
#        mfc_tar = pw.code_spectral_envelope(sp_tar, 16000, 23)
#        
#        src_mfcc = librosa.feature.mfcc(y=src, sr=sample_rate, \
#                                        hop_length=int(sample_rate*window_len), \
#                                        win_length=int(sample_rate*window_len), \
#                                        n_fft=1024, n_mels=128)
#            
#        tar_mfcc = librosa.feature.mfcc(y=tar, sr=sample_rate, \
#                                        hop_length=int(sample_rate*window_len), \
#                                        win_length=int(sample_rate*window_len), \
#                                        n_fft=1024, n_mels=128)
#
#        _, cords = librosa.sequence.dtw(X=src_mfcc, Y=tar_mfcc, metric='cosine')
#        cords = np.flipud(cords)
#        sp_src = sp_src[cords[:,0],:]
#        sp_tar = sp_tar[cords[:,1],:]
        
        
#        for i in range(10):
#            q = np.random.randint(0, len(cords))
#            pylab.figure(), pylab.subplot(211)
#            pylab.plot(sp_src[cords[q,0],:], label='neutral')
#            pylab.plot(sp_tar[cords[q,1],:], label='angry')
#            pylab.grid(), pylab.title('Slice %d' % q), pylab.legend(loc=1)
#            
#            pylab.subplot(212)
#            pylab.plot(mfc_src[cords[q,0],:], label='neutral')
#            pylab.plot(mfc_tar[cords[q,1],:], label='angry')
#            pylab.grid(), pylab.title('Slice %d' % q), pylab.legend(loc=1)
        
#        u_src, sigma_src, v_src = np.linalg.svd(sp_src)
#        u_tar, sigma_tar, v_tar = np.linalg.svd(sp_tar)      
#            
#        s_mat = np.zeros(sp_src.shape)
#        t_mat = np.zeros(sp_tar.shape)
#        s_mat_array = []
#        t_mat_array = []
#        for i in range(min([u_src.shape[0], v_src.shape[0]])):
#            x = np.dot(u_src[:,i:i+1], v_src[i:i+1,:])
#            s_mat += sigma_src[i]*x
#            s_mat_array.append(s_mat)
#            pylab.figure(figsize=(15,15)), pylab.imshow(_power_to_db(s_mat.T ** 2))
#            pylab.suptitle('#Components %d' % (i+1))
#            pylab.savefig('/home/ravi/Desktop/svd_recon/src_'+str(i)+'.png')
#            pylab.close()
#            
#        for i in range(min([u_tar.shape[0], v_tar.shape[0]])):
#            y = np.dot(u_tar[:,i:i+1], v_tar[i:i+1,:])
#            t_mat += sigma_tar[i]*y
#            t_mat_array.append(t_mat)
#            pylab.figure(figsize=(15,15)), pylab.imshow(_power_to_db(s_mat.T ** 2))
#            pylab.suptitle('#Components %d' % (i+1))
#            pylab.savefig('/home/ravi/Desktop/svd_recon/tar_'+str(i)+'.png')
#            pylab.close()
#        
#        break
            
#        s_mfc_array = np.asarray([pw.code_spectral_envelope(s, 16000, 4) for s in s_mat_array])
#        t_mfc_array = np.asarray([pw.code_spectral_envelope(t, 16000, 4) for t in t_mat_array])
#        
#        print(w)
#        min_val.append((np.min(s_mfc_array) ,np.min(t_mfc_array)))
#        max_val.append((np.max(s_mfc_array) ,np.max(t_mfc_array)))


    """
    Cohort analysis
    """        
    src_list = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/neutral-angry/train/neutral', '*.wav')))
    tar_list = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/neutral-angry/train/angry', '*.wav')))
    
    executor = ProcessPoolExecutor(max_workers=8)
    src_futures = []
    tar_futures = []
    
    src_results = []
    tar_results = []
    
    dim = 8
    sampling = 2
    for sampling in range(sampling):
        for i in src_list:
            src_futures.append(executor.submit(partial(_get_spect_no_abs, i, dim, False)))
    #        src_results.append(_get_spect_no_abs(i, dim))
    #        print(i)
    src_results = [src_future.result() for src_future in tqdm(src_futures)]
    
    for sampling in range(sampling):
        for i in tar_list:
            tar_futures.append(executor.submit(partial(_get_spect_no_abs, i, dim, False)))
    #        tar_results.append(_get_spect_no_abs(i, dim))
    #        print(i)
    tar_results = [tar_future.result() for tar_future in tqdm(tar_futures)]

    src_mfcc = [i for i,j in zip(src_results, tar_results) if i!=None and j!=None]
    tar_mfcc = [j for i,j in zip(src_results, tar_results) if i!=None and j!=None]

    src_rank1 = np.asarray([i[0] for i in src_mfcc])
    src_rank2 = np.asarray([i[1] for i in src_mfcc])
    src_rank3 = np.asarray([i[2] for i in src_mfcc])
    src_rank4 = np.asarray([i[3] for i in src_mfcc])
    src_rank5 = np.asarray([i[4] for i in src_mfcc])
    src_rank6 = np.asarray([i[5] for i in src_mfcc])
    src_rank7 = np.asarray([i[6] for i in src_mfcc])
    src_rank8 = np.asarray([i[7] for i in src_mfcc])

    tar_rank1 = np.asarray([i[0] for i in tar_mfcc])
    tar_rank2 = np.asarray([i[1] for i in tar_mfcc])
    tar_rank3 = np.asarray([i[2] for i in tar_mfcc])
    tar_rank4 = np.asarray([i[3] for i in tar_mfcc])
    tar_rank5 = np.asarray([i[4] for i in tar_mfcc])
    tar_rank6 = np.asarray([i[5] for i in tar_mfcc])
    tar_rank7 = np.asarray([i[6] for i in tar_mfcc])
    tar_rank8 = np.asarray([i[7] for i in tar_mfcc])
    
    src_ranks = [src_rank1, src_rank2, src_rank3, src_rank4, src_rank5, src_rank6, src_rank7, src_rank8]
    tar_ranks = [tar_rank1, tar_rank2, tar_rank3, tar_rank4, tar_rank5, tar_rank6, tar_rank7, tar_rank8]
#
#    n_data = src_rank1.shape[0]
#    kl_div = []
#    norm_v = []
#    for i in range(8):
#        try:
#            tsne = TSNE(n_components=2, n_iter=2000, verbose=True)
#            embed_rank = tsne.fit_transform(np.concatenate((src_ranks[i].reshape(-1,128*dim), 
#                                                             tar_ranks[i].reshape(-1,128*dim)), 
#                                                                axis=0))
#            norm_v.append(np.linalg.norm(np.mean(embed_rank[:n_data]) - np.mean(embed_rank[n_data:])))
#            kl_div.append(tsne.kl_divergence_)
#            pylab.figure()
#            pylab.plot(embed_rank[:n_data,0], embed_rank[:n_data,1], 'r.')
#            pylab.plot(embed_rank[n_data:,0], embed_rank[n_data:,1], 'b.')
#            pylab.title('Rank %d' % (i+1))
#            print('######################## Norm is %f ############' % norm_v[-1])
#        except Exception as ex:
#            print(ex)
    
    
    
    
    
    






    
    
    