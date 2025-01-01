## Inference python scripts
import os
import argparse
import logging
import torch
import torch.multiprocessing as mp
import tqdm
import time
from pathlib import Path

import torchaudio
import soundfile as sf
from utils.utils import AttrDict, update_args, setup_seed
from bin.se_inference import SpeechEnhancement
from utils.utils import get_source_list
from utils.mel_spectrogram import MelSpec, rms_normalize


def parse_args():
    parser = argparse.ArgumentParser()
    ## laura gpt related
    parser.add_argument("--sampling", default = 25, type = int)
    parser.add_argument("--beam_size", default = 1, type = int)

    parser.add_argument("--scp", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--output_dir", type=str)
    ## DDP
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(1234, 0)
    mp.spawn(inference, args=(args,), nprocs=args.num_proc, join=True)
    print("done!")


def inference(rank, args):
    # update args to contain config
    update_args(args, args.config)
    args = AttrDict(**vars(args))
    args.output_dir = Path(args.output_dir)
    print(f"args: {args}")
    # device setup
    device = args.gpus[rank % len(args.gpus)]
    # data for each process setup
    scp = get_source_list(args.scp)
    scp = scp[rank :: args.num_proc]
    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # load model
    sp_en = SpeechEnhancement(args, args.model_ckpt, device, logger)
    # mel spec
    mel_spec = MelSpec(normalization=args.norm_noisy)

    # Inference
    total_rtf = 0.0
    with torch.no_grad(), tqdm.tqdm(scp, desc="[inferencing...]") as pbar:
        for audio_path in pbar:
            # 0. Load audio & Extract Mel
            audio, sr = torchaudio.load(audio_path)  # [1,T]
            mask = torch.tensor([audio.size(1)], dtype=torch.long)
            mel, _ = mel_spec.mel(audio, mask)
            mel = mel.to(device)

            # 1. Inference
            start = time.time()
            output = sp_en(mel)[0]["gen"].squeeze()  # [T]
            rtf = (time.time() - start) / (len(output) / sr)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # 2. Save audio
            base_name = Path(audio_path).stem + ".wav"
            save_path = args.output_dir / base_name
            
        
            sf.write(save_path, rms_normalize(output.cpu().numpy()), samplerate=sr)
    logger.info(f"Finished generation of {len(scp)} utterances (RTF = {total_rtf / len(scp):.03f}).")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # parser = argparse.ArgumentParser()
    # ## Codec
    # parser.add_argument("--codec_config_file", type=str)
    # parser.add_argument("--codec_model_file", type=str)
    #
    # parser.add_argument("--config", type=str)
    # parser.add_argument("--model_ckpt", type=str)
    # parser.add_argument("--output_dir", type=str)
    # parser.add_argument("--raw_inputs", nargs="*", default=None, type=str)
    # parser.add_argument("--tokenize_to_phone", action="store_true")
    # args = parser.parse_args()
    # update_args(args, args.default_config)
    # main(args)
    # pass
