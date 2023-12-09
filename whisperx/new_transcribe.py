import gc
import os
import warnings

import numpy as np
import torch

from .alignment import align, load_align_model
from .asr import load_model
from .audio import load_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .utils import (LANGUAGES, TO_LANGUAGE_CODE, get_writer, optional_float,
                    optional_int, str2bool)



def new_cli(audio_path,model="large-v2",_language="en",_align_model="WAV2VEC2_ASR_LARGE_LV60K_960H",
        _diarize=False,_min_speakers=None,_max_speakers=None,_hf_token=None,_print_progress=True):{

    model_name = model
    device = "cuda"
    device_index=0
    batch_size=8
    compute_type="float16"
    
    output_dir="."
    output_format="srt"
    verbose=True

    task="transcribe"
    language= _language


    # alignment params
    align_model = _align_model
    interpolate_method = "nearest"
    no_align = False
    return_char_alignments = False

    # vad params
    vad_onset = 0.100
    vad_offset = 0.363
    chunk_size = 30
    
    # diarization params
    diarize = _diarize
    min_speakers = _min_speakers
    max_speakers = _max_speakers
    
    temperature=0
    best_of=5
    beam_size=5
    patience=1.0
    length_penalty=1.0
    suppress_tokens="-1"
    suppress_numerals=False
    initial_prompt=None
    condition_on_previous_text=False
    fp16=True

    temperature_increment_on_fallback = 0.2
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold = 0.6

    max_line_width = None
    max_line_count = None
    highlight_words = False
    segment_resolution = "sentence"

    threads = 0
    hf_token = _hf_token
    print_progress= _print_progress
    align_language = _language
    

    os.makedirs(output_dir, exist_ok=True)

    if (increment := temperature_increment_on_fallback) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if (threads := threads) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads
  
    asr_options={
        "beam_size": beam_size,
        "patience": patience,
        "length_penalty": length_penalty,
        "temperatures": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": initial_prompt,
        "suppress_tokens": [int(x) for x in suppress_tokens.split(",")],
        "suppress_numerals": suppress_numerals,
    }

    writer = get_writer(output_format, output_dir)
    # word_options = ["highlight_words", "max_line_count", "max_line_width"]
    # if no_align:
    #     for option in word_options:
    #         if args[option]:
    #             parser.error(f"--{option} not possible with --no_align")
    # if args["max_line_count"] and not args["max_line_width"]:
    #     warnings.warn("--max_line_count has no effect without --max_line_width")
    # writer_args = {arg: args.pop(arg) for arg in word_options}

    # Part 1: VAD & ASR Loop
    results = []
    tmp_results = []
    # model = load_model(model_name, device=device, download_root=model_dir)
    model = load_model(model_name, device=device, device_index=device_index, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)

    #for audio_path in args.pop("audio"):
    audio = load_audio(audio_path)
    # >> VAD & ASR
    print(">>Performing transcription...")
    result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size, print_progress=print_progress)
    results.append((result, audio_path))
    print(results)

    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()


    for result, audio_path in results:
        result["language"] = align_language
        writer(result, audio_path)
   

}

if __name__ == "__main__":
    new_cli()