username=yhchen70  #username
proID=2227  # dlp project id
gpuType=TeslaV100-PCIE-32GB # TeslaV100-PCIE-12GB
reserved=dlp3-tts-prompt-reserved # public: tts, reserved: dlp3-tts-reserved, dlp3-tts-project-reserved, dlp3-tts-prompt-reserved, cdlp3-tts-industry-reserved dlp3-tts-prompt-ml-reserved

log_dir=/yrfs5/tts/yhchen70/projects/phoneme_asr/logs
mkdir -p $log_dir

# ===== 修改区 =====
wav_dir="/yrfs5/tts/yhchen70/projects/phoneme_asr/input_wavs/test50"
text_txt="/yrfs5/tts/yhchen70/projects/phoneme_asr/input_txt/test50"
output_txt="outputs/test50_asr_result.txt"
device="cuda"
root_src=/yrfs5/tts/yhchen70/projects/phoneme_asr
# ======================================================================




submit_dir=$logdir/submit
mkdir -p $submit_dir

echo ". $root_src/run_huper_tsv.sh \
    "$wav_dir" \
    "$text_txt" \
    "$output_txt" \
    "$device"
" > $submit_dir/asr.sh
# bash $submit_dir/pydur${i}.sh

ky exp submit -a $username -n data-asr-enus$i -d asr$i  -i reg.deeplearning.cn/ky/ubuntu-kaldi-image:latest -e $submit_dir/asr.sh --useGpu -g 1 -k $gpuType -t PtJob -l $log_dir/asr.log -o $log_dir/asr_error.log     --proID $proID    --modelPath $log_dir --modelName $(date "+%Y%m%d%H%M%S")  -r ${reserved}
