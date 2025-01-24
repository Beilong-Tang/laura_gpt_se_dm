import argparse 
import soundfile as sf 
import sys 
import os
from pathlib import Path
import tqdm as tqdm
import torch.multiprocessing as mp
import librosa
sys.path.append(os.getcwd())

from utils.utils import get_source_list, list_to_files, merge_content

SEED = 1234

def read_audio(filename, force_1ch=False, fs=None):
    """
    Return audio of shape [channel, frame]
    if `force_1ch=True`, it will be mono-channel regardless of the original audio
    """
    audio, fs_ = sf.read(filename, always_2d=True)
    audio = audio[:, :1].T if force_1ch else audio.T
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
        return audio, fs
    return audio, fs_


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scp", type = str, required = True)
    p.add_argument("--base_path", type = str, required = True)
    p.add_argument("--output_dir", type = str, required = True)
    p.add_argument("--sample_rate", type = int, required = True)

    p.add_argument("--num_proc", type = int, default = 4)
    return p.parse_args()

def main(args):
    # names, scp_paths = get_source_list(args.scp, ret_name= True)
    if args.num_proc > 1:
        mp.spawn(run, args = (args, ), nprocs=args.num_proc, join= True)
    elif args.num_proc == 1:
        run(0, args)
    else:
        raise Exception("Invalid args.num_proc. It should be greater than or equal to 1.")
    
    ## Merge scp to one 
    scp_paths = [str(Path(args.output_dir) / f"temp_sr_{args.sample_rate}_rank_{i}.scp") for i in range(args.num_proc)]
    merge_content(scp_paths, save_path = str(Path(args.output_dir) / f"all_sr_{args.sample_rate}.scp"))

    print("Done...")

def run(rank, args):
    names, scp_paths = get_source_list(args.scp, ret_name= True)
    names = names[rank::args.num_proc]
    scp_paths = scp_paths[rank::args.num_proc]

    scp_lines = []
    with tqdm.tqdm(list(zip(names, scp_paths)), desc=f"[rank {rank}]") as pbar:
        for _name, _scp in pbar:

            # Create absolute path 
            abs_scp = str(Path(args.base_path).absolute() / _scp) 
            
            # Resample Audio 
            audio = read_audio(abs_scp, True, args.sample_rate)[0].squeeze(0) # [T]

            # Output Directory
            save_dir = Path(args.output_dir).absolute() / f'wavs_sr_{args.sample_rate}' / Path(_scp).parent
            os.makedirs(save_dir, exist_ok= True)
            save_path = str(save_dir / (Path(_scp).stem + ".wav"))

            # Save Audio
            sf.write(save_path, audio, samplerate=args.sample_rate)

            # Add it to scp to save
            scp_lines.append(f"{_name} {save_path}\n")
    
    list_to_files(scp_lines, str(Path(args.output_dir) / f"temp_sr_{args.sample_rate}_rank_{rank}.scp"))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass
