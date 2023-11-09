#!/bin/bash
#SBATCH --job-name=hmsc-hpc_fit
#SBATCH --account=project_462000235
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=01:14:59 --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --array=0-7

ind=301
MT=${1:-0}
SAM=${2:-100}
THIN=${3:-10}
PROFILE=${4:-0}

module use /appl/local/csc/modulefiles
module load tensorflow/2.12
export PYTHONPATH=$PWD/../../hmsc-hpc:$PYTHONPATH
echo $PYTHONPATH
hostname

modelTypeStringSuffices=("ns" "fu" "pg" "nn" "ph")
modelTypeString="$MT${modelTypeStringSuffices[$MT]}"

nsVec=(10 20 40 80 160 320 622)
nyVec=(100 200 400 800 1600 3200 6400 12800 25955 51910 103820 207640)

ns=${nsVec[$(($ind / 100 - 1))]}
ny=${nyVec[$(($ind % 100 - 1))]}
nChains=8

data_path="/scratch/project_462000235/gtikhono/lumiproj_2022.06.03_HPC_development/examples/big_spatial"
input_path=$data_path/$(printf "init/init_%s_ns%.3d_ny%.5d_chain%.2d.rds" $modelTypeString $ns $ny $nChains)
output_path=$data_path/$(printf "fmTF/TF_%s_ns%.3d_ny%.5d_chain%.2d_sam%.4d_thin%.4d_c%.2d.rds" $modelTypeString $ns $ny $nChains $SAM $THIN $SLURM_ARRAY_TASK_ID)

srun python3 ./../../hmsc-hpc/hmsc/examples/run_gibbs_sampler.py --input $input_path --output $output_path --samples $SAM --transient $(($SAM*$THIN)) --thin $THIN --verbose 100 --chain $SLURM_ARRAY_TASK_ID --fse 0  --profile $PROFILE
