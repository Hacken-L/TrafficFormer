import os
import subprocess
import sys
import torch

from pretrain_data_gen import pretrain_dataset_generation, corpora_to_bigram
from vocab_gen import build_BPE, build_vocab
from finetuning_data_gen import convert_splitcap, generation_multiP, dataset_extract, enhance_based_tsv

repo_root = "/data/TrafficFormer_1.0"
pretrain_output = "/data/TrafficFormer_1.0/data_generation/pretrain_output/"
tmp_burst = os.path.join(pretrain_output, "tmp_burst/")
corpora_path = os.path.normpath(tmp_burst.rstrip("/")) + "_biburst.txt"
corpora_bigram_path = os.path.join(pretrain_output, "tmp_burst_biburst_bigram.txt")
tokenizer_json_path = os.path.join(pretrain_output, "wordpiece.tokenizer.json")
vocab_path = os.path.join(pretrain_output, "encryptd_vocab.txt")
dataset_pt_path = os.path.join(pretrain_output, "dataset.pt")
output_pretrain_model_path = os.path.join(pretrain_output, "model.bin")
finetune_output = os.path.join(repo_root, "data_generation", "finetune_output") + os.sep
preprocess_script = os.path.join(repo_root, "pre-training", "preprocess.py")
pretrain_script = os.path.join(repo_root, "pre-training", "pretrain.py")
finetune_script = os.path.join(repo_root, "fine-tuning", "run_classifier.py")
finetuned_model_path = os.path.join(repo_root, "models", "finetuned_model.bin")



def run_model_pretrain(
    *,
    use_cpu=False,
    cuda_visible_devices=None,
    world_size=1,
    gpu_ranks=(0,),
    master_ip="tcp://localhost:12345",
    total_steps=90,
    save_checkpoint_steps=10,
    batch_size=32,
):

    """CPU 时默认压低 instances_buffer_size"""
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    try:
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    cmd = [
        sys.executable,
        pretrain_script,
        "--dataset_path",
        dataset_pt_path,
        "--vocab_path",
        vocab_path,
        "--output_model_path",
        output_pretrain_model_path,
        "--master_ip",
        master_ip,
        "--total_steps",
        str(total_steps),
        "--save_checkpoint_steps",
        str(save_checkpoint_steps),
        "--batch_size",
        str(batch_size),
        "--embedding",
        "word_pos_seg",
        "--encoder",
        "transformer",
        "--mask",
        "fully_visible",
        "--target",
        "bertflow",
    ]

    if use_cpu or not has_cuda:
        cmd.extend(["--world_size", "1", "--instances_buffer_size", "8192"])
    else:
        cmd.extend(["--world_size", str(world_size), "--gpu_ranks", *[str(r) for r in gpu_ranks]])

    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def run_model_finetune(
    *,
    cuda_visible_devices=None,
    train_path=None,
    dev_path=None,
    test_path=None,
    pretrained_model_path=None,
    finetune_output_path=None,
    finetune_vocab_path=None,
    epochs_num=4,
    earlystop=4,
    batch_size=128,
    seq_length=320,
    learning_rate=6e-5,
):
    """README「Model Finetuning」: 需已有预训练权重 ``pretrain_output/model.bin``（需要先执行 ``run_model_pretrain``）。"""
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    train_path = train_path or os.path.join(finetune_output, "dataset", "train_dataset.tsv")
    dev_path = dev_path or os.path.join(finetune_output, "dataset", "valid_dataset.tsv")
    test_path = test_path or os.path.join(finetune_output, "dataset", "test_dataset.tsv")
    pretrained_model_path = output_pretrain_model_path
    finetune_output_path = finetune_output_path or finetuned_model_path
    finetune_vocab_path = finetune_vocab_path or vocab_path

    os.makedirs(os.path.dirname(finetune_output_path), exist_ok=True)

    cmd = [
        sys.executable,
        finetune_script,
        "--vocab_path",
        finetune_vocab_path,
        "--train_path",
        train_path,
        "--dev_path",
        dev_path,
        "--test_path",
        test_path,
        "--pretrained_model_path",
        pretrained_model_path,
        "--output_model_path",
        finetune_output_path,
        "--epochs_num",
        str(epochs_num),
        "--earlystop",
        str(earlystop),
        "--batch_size",
        str(batch_size),
        "--embedding",
        "word_pos_seg",
        "--encoder",
        "transformer",
        "--mask",
        "fully_visible",
        "--seq_length",
        str(seq_length),
        "--learning_rate",
        str(learning_rate),
    ]
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    
    pretrain_dataset_generation(
        pcapng_path="/data/TrafficFormer_1.0/data_generation/pcap_pretrain/",
        pcap_output_path="/data/TrafficFormer_1.0/data_generation/pcap_pretrain/",
        output_split_path=pretrain_output,
        select_packet_len=80,
        corpora_path=tmp_burst,
        start_index=28,
        enhance_factor=1,
        is_multi=True,
    )

    
    
    corpora_to_bigram(corpora_path, corpora_bigram_path)
    build_BPE(corpora_bigram_path, tokenizer_json_path=tokenizer_json_path)
    build_vocab(vocab_path, tokenizer_json_path=tokenizer_json_path)
    
    
    

    # README「Pretrain Input Generation」: 用语料 + 词表生成 bertflow 预训练用的 dataset.pt（多进程切分、按流组 batch 等逻辑在 uer 内）
    subprocess.run(
        [
            sys.executable,
            preprocess_script,
            "--corpus_path",
            corpora_bigram_path,
            "--vocab_path",
            vocab_path,
            "--seq_length",
            "512",
            "--dataset_path",
            dataset_pt_path,
            "--processes_num",
            "80",
            "--target",
            "bertflow",
        ],
        cwd=repo_root,
        check=True,
    )
    
    # README「Model Pretrain」: 耗时长。多卡: run_model_pretrain(cuda_visible_devices="2,3,4", world_size=3, gpu_ranks=(0,1,2))
    # 强制 CPU 仍 OOM 时: run_model_pretrain(use_cpu=True, batch_size=8)
    run_model_pretrain()


    # finetune
    # pcapng_path = os.path.join(repo_root, "data_generation", "pcapng") + os.sep
    pcap_path = os.path.join(repo_root, "data_generation", "pcap_finetune") + os.sep
    pcap_split_path = os.path.join(repo_root, "data_generation", "pcap_split_finetune") + os.sep

    convert_splitcap(pcap_path, pcap_path, pcap_split_path)
    # 如果每个子目录一类，每个子目录一个文件夹的话: convert_splitcap(..., is_pcap_label=True)

    # README「Generate Data」: pcap → dataset.json；
    samples_per_class = 100
    _category = 10
    # dataset_save_path = os.path.join(finetune_output,, "finetune_dataset")
    os.makedirs(finetune_output, exist_ok=True)
    generation_multiP(
        pcap_split_path + "splitcap" + "/",
        [samples_per_class] * _category,
        finetune_output,
        start_index=28,
    )

    # README「From dataset.json to train/valid/test tsv」: 写到 finetune_output/dataset/*.tsv
    dataset_extract(
        finetune_output,
        ["datagram", "length", "time", "direction", "message_type"],
    )

    # README「Data Augmentation」: 生成 train_enhance5_dataset.tsv
    enhance_based_tsv(
        finetune_output + "dataset/",
        "train_dataset.tsv",
        "train_enhance5",
        enhance_factor=5,
    )

    # README「Model Finetuning」:使用 GPU 2 时: run_model_finetune(cuda_visible_devices="2")
    run_model_finetune()
    