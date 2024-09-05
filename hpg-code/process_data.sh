#!/bin/bash
#SBATCH --job-name=preprocess_emotion_attack  # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=akshayashok@ufl.edu    # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks=2                    # Run on a single CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/preprocess_emotion_attack_%j.log   # Standard output and error log
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang


module load conda
conda activate emotion

cd /blue/ruogu.fang/akshayashok/emotion_adversarial_attack/code

python process_data.py --dataset emoset
python process_data.py --dataset abstract
python process_data.py --dataset artphoto
python process_data.py --dataset CAER-S
python process_data.py --dataset D-ViSA
python process_data.py --dataset FI

date;hostname;pwd
