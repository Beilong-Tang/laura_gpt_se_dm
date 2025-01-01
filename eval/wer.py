"""
Command line tool to output the transcription of a list of audio files.
"""

import whisper
import argparse
import os.path as op
import glob
import tqdm
import os
import sys 
sys.path.append(os.getcwd())
import torch
import torch.multiprocessing as mp


def main(args, device, rank, world_size):
    torch.cuda.set_device(device)
    model = whisper.load_model(args.model)
    model.cuda()
    audio_path_list = sorted(glob.glob(op.join(args.test_file, "*.wav")))
    audio_path_list = audio_path_list[rank::world_size]
    print("total audio len: ", len(audio_path_list))
    os.makedirs(op.dirname(args.output), exist_ok=True)
    res = []
    for audio in tqdm.tqdm(audio_path_list):
        result = model.transcribe(audio)
        res.append(f"{os.path.basename(audio)}|{result['text']}\n")
    with open(f"{args.output}.temp_{rank}.txt", "w") as f:
        f.writelines(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test_file",
        help="The path of audio files ending with .wav to be transcripted",
    )
    parser.add_argument(
        "-o", "--output", help="The output file containing the transcript"
    )
    parser.add_argument(
        "-m", "--model", help="The model to use, default: base", default="base"
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()
    
    processes = []
    for i in range(0,args.num_proc):
        device = args.gpus[i % len(args.gpus)]
        process = mp.Process(target=main, args=(args, device, i, args.num_proc))
        process.start()
        processes.append(process)
    for p in processes:
        p.join()
    results = []
    for path in glob.glob(f"{args.output}.temp*.txt"):
        with open(path, "r") as f:
            for line in f.readlines():
                results.append(line)
        os.remove(path)
    with open(args.output,"w") as file:
        file.writelines(results)

