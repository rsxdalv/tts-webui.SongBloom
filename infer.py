import glob
import time
import torch
import re
import torch.nn.functional as F

from itertools import chain
import torchaudio
from tqdm import tqdm
import json
import typing as tp
import numpy as np
import time

from allprompt1129.dataset_2min import TestData

import os, sys

os.environ['DISABLE_FLASH_ATTN'] = "1"
from SongBloom.registry import load_config
from SongBloom.datasets.lyric_common import key2processor, symbols, LABELS

from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler




def main():
    cfg = load_config()
  
    cfg.max_dur = 150
    model = SongBloom_Sampler.build_from_trainer(cfg, strict=True)
    
    N_sample = 2
    ckpt_name = [i for i in cfg.pretrained_path.split("/") if i.startswith("step=")][0]
    save_dir = f"output_150s/{os.path.splitext(os.path.basename(sys.argv[1]))[0]}/{ckpt_name}/cfg{cfg.inference.cfg_coef}_{cfg.inference.dit_cfg_type}_step{cfg.inference.steps}" #+ (f"_steps{cfg.inference.steps}" if cfg.inference.get("steps", "") != "" else "")
                
    os.makedirs(save_dir, exist_ok=True)
    
    # model.eval()
    model.set_generation_params(use_sampling=True, **cfg.inference)  
    # import pdb; pdb.set_trace()

    lyric_processor = key2processor.get(cfg.train_dataset.lyric_processor) if cfg.train_dataset.lyric_processor is not None else lambda x: x
    lyric_processor_key = cfg.train_dataset.lyric_processor
    def _process_lyric(input_lyric):
        if lyric_processor_key == 'pinyin':
            processed_lyric = lyric_processor(input_lyric)
        else:
            processed_lyric = []
            check_lyric = input_lyric.split(" ")
            for ii in range(len(check_lyric)):
                if check_lyric[ii] not in symbols and check_lyric[ii] not in LABELS.keys() and len(check_lyric[ii]) > 0:
                    new = lyric_processor(check_lyric[ii])
                    check_lyric[ii] = new
            processed_lyric = " ".join(check_lyric)
        
        print(input_lyric)
        print(processed_lyric)
        return processed_lyric
        
    # save_lrc = open("out_lrc.txt", 'w')
    time_consuming = 0.
    gen_length = 0.
    
    for idx, test_sample in enumerate(TestData()):
        # print(test_sample)
        genre, lyrics, prompt_wav = test_sample 
        # genre = genre[0]
        generate_inp = {
            "lyrics": [_process_lyric(lyrics[0])],
            'prompt_wav': [prompt_wav[0].mean(dim=0, keepdim=True)],
        }
        print(generate_inp)
        print(generate_inp['prompt_wav'][0].shape)
        # breakpoint()
        for i in range(N_sample):

            if os.path.exists(f'{save_dir}/{genre}_s{i}.flac'):
                continue
            st = time.time()
            wav = model.generate(**generate_inp)
            time_consuming += time.time() - st
            gen_length += wav.shape[-1] / 48000
            os.makedirs(os.path.dirname(f'{save_dir}/{genre}_s{i}.flac'), exist_ok=True)
            torchaudio.save(f'{save_dir}/{genre}_s{i}.flac', wav[0].cpu(), model.sample_rate)
            print(f'{save_dir}/{genre}_s{i}.flac')

    print("RTF:\t", time_consuming / (gen_length + 1e-9))
if __name__ == "__main__":
    
    main()
