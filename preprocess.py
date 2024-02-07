def prepare_mc_wsj_csv(
    datapath,
    savepath,
    skip_prep=False,
    fs=8000,
    vocab="20k",
    array=1,
    array_ch=1
):
    if skip_prep:
        return
    datapath = os.path.join(datapath, "data/audio")
    processed_data_path = os.path.join(datapath, f"{vocab}_aligned_{str(fs).replace('000','k')}")

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
        align_files = True
    else:
        align_files = False
    
    if align_files:
        mix_root = os.path.join(datapath,"olap")
        mix_paths = glob.glob(os.path.join(mix_root, f"*/array{array}/{vocab}/*Array{array}-{array_ch}*.flac"))
        s1_paths = [
            mix.replace(f"array{array}", "headset1").replace(f"Array{array}-{array_ch}", "Headset1") 
            for mix in mix_paths
            ]
        s2_paths = [
            mix.replace(f"array{array}", "headset2").replace(f"Array{array}-{array_ch}", "Headset2")
            for mix in mix_paths
            ]
        
        # create new directory for aligned data
        new_mix_root = os.path.join(processed_data_path, "mix")
        new_s1_root = os.path.join(processed_data_path, "s1")
        new_s2_root = os.path.join(processed_data_path, "s2")
        os.makedirs(new_mix_root)
        os.makedirs(new_s1_root)
        os.makedirs(new_s2_root)

        remove_idxs = []

        for i, (mix_path, s1_path, s2_path) in tqdm(enumerate(zip(mix_paths, s1_paths, s2_paths)), total=len(mix_paths)):
            assert os.path.exists(mix_path), f"{mix_path} does not exist"

            try:
                assert os.path.exists(s1_path), f"{s1_path} does not exist"
            except AssertionError as e:
                if "_x_" in mix_path:
                    s1_path = s1_path.replace("_x_", "_")
                    print(f"Removed _x_ from {mix_path} to get {s1_path}")
                    if not os.path.exists(s1_path):
                        print(f"Skipping {mix_path} because {s1_path} does not exist")
                        # Remove index from lists
                        remove_idxs.append(i)
                        continue
                else:
                    raise e
                
            try:
                assert os.path.exists(s2_path), f"{s2_path} does not exist"
            except AssertionError as e:
                if "_x_" in mix_path:
                    s2_path = s2_path.replace("_x_", "_")
                    print(f"Removed _x_ from {mix_path} to get {s2_path}")
                    if not os.path.exists(s1_path):
                        print(f"Skipping {mix_path} because {s1_path} does not exist")
                        # Remove index from lists
                        remove_idxs.append(i)
                        continue
                else:
                    raise e
                

            # Load the audio
            mix_sig, _ = sf.read(mix_path, dtype="float32")
            s1_sig, _ = sf.read(s1_path, dtype="float32")
            s2_sig, orig_fs = sf.read(s2_path, dtype="float32")

            # Change loudness of mix_sig to match s1_sig and s2_sig
            mix_loudness = pyln.Meter(fs).integrated_loudness(mix_sig)

            if len(s1_sig) < len(s2_sig):
                clean_mix = s1_sig + s2_sig[:len(s1_sig)]
            else:
                clean_mix = s1_sig[:len(s2_sig)] + s2_sig
            clean_mix_loudness = pyln.Meter(fs).integrated_loudness(clean_mix)
            mix_sig = pyln.normalize.loudness(mix_sig, mix_loudness, clean_mix_loudness)


            # Decimate from orig_fs to fs
            if orig_fs != fs:
                mix_sig = decimate(mix_sig, orig_fs // fs)
                s1_sig = decimate(s1_sig, orig_fs // fs)
                s2_sig = decimate(s2_sig, orig_fs // fs)

            # Align s1_sig and s2_sig to mix_sig using cross-correlation
            s1_xcorr = correlate(s1_sig, mix_sig, mode="full")
            s2_xcorr = correlate(mix_sig, s2_sig, mode="full")
            s1_delta = s1_xcorr.argmax() - len(s1_sig) 
            s2_delta = s2_xcorr.argmax() - len(s2_sig) 

            if s1_delta >=0:
                s1_sig = s1_sig[s1_delta:]
            else:
                s1_Sig = s1_sig[:s1_delta]
            if s2_delta >=0:
                s2_sig = s2_sig[s2_delta:]
            else:
                s2_sig = s2_sig[:s2_delta]

            min_len = min(len(mix_sig), len(s1_sig), len(s2_sig))
            mix_sig = mix_sig[:min_len]
            s1_sig = s1_sig[:min_len]
            s2_sig = s2_sig[:min_len]

            # Save the aligned audio
            mix_path = os.path.join(new_mix_root, os.path.basename(mix_path).replace(".flac",".wav"))
            s1_path = os.path.join(new_s1_root, os.path.basename(mix_path).replace(".flac",".wav"))
            s2_path = os.path.join(new_s2_root, os.path.basename(mix_path).replace(".flac",".wav"))
            sf.write(mix_path, mix_sig, fs)
            sf.write(s1_path, s1_sig, fs)
            sf.write(s2_path, s2_sig, fs)

            # Update the paths
            mix_paths[i] = mix_path
            s1_paths[i] = s1_path
            s2_paths[i] = s2_path
        # Remove dodgy indexes from lists
        for i in sorted(remove_idxs, reverse=True):
            mix_paths.pop(i)
            s1_paths.pop(i)
            s2_paths.pop(i)
    else:
        mix_paths = glob.glob(os.path.join(processed_data_path, "mix", "*.wav"))
        if len(mix_paths) == 0:
            raise FileNotFoundError(f"No mix files found in {processed_data_path}")
        s1_paths = [mix_path.replace("mix", "s1") for mix_path in mix_paths]
        s2_paths = [mix_path.replace("mix", "s2") for mix_path in mix_paths]

        # Assert s1 and s2 paths exist
        for s1_path, s2_path in tqdm(zip(s1_paths, s2_paths)):
            assert os.path.exists(s1_path), f"{s1_path} does not exist"
            assert os.path.exists(s2_path), f"{s2_path} does not exist"

    

    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
        "noise_wav",
        "noise_wav_format",
        "noise_wav_opts",
    ]

    with open(savepath + "/mc_wsj_" + vocab + ".csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        
        for i, (mix_path, s1_path, s2_path) in tqdm(enumerate(zip(mix_paths, s1_paths, s2_paths)), total=len(mix_paths)):
            row = {
                "ID": i,
                "duration": 1.0,
                "mix_wav": mix_path,
                "mix_wav_format": "wav",
                "mix_wav_opts": None,
                "s1_wav": s1_path,
                "s1_wav_format": "wav",
                "s1_wav_opts": None,
                "s2_wav": s2_path,
                "s2_wav_format": "wav",
                "s2_wav_opts": None,
                "noise_wav": mix_path, # dummy to supress errors
                "noise_wav_format": None,
                "noise_wav_opts": None,
            }
            writer.writerow(row)
