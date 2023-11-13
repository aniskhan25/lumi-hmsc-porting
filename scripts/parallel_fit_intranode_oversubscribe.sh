#!/bin/bash
#SBATCH --job-name=hmsc-hpc_fit
#SBATCH --account=project_462000235
#SBATCH --output=output/%A
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --time=00:15:00

time0=$(date +%s.%N)

ind=301
MT=${1:-0}
SAM=${2:-100}
THIN=${3:-10}
PROFILE=${4:-0}

source $WORKDIR/setup-env.sh
mkdir -p output

modelTypeStringSuffices=("ns" "fu" "pg" "nn" "ph")
modelTypeString="$MT${modelTypeStringSuffices[$MT]}"

nsVec=(10 20 40 80 160 320 622)
nyVec=(100 200 400 800 1600 3200 6400 12800 25955 51910 103820 207640)

ns=${nsVec[$(($ind / 100 - 1))]}
ny=${nyVec[$(($ind % 100 - 1))]}
nChains=8

data_path="/scratch/project_462000235/gtikhono/lumiproj_2022.06.03_HPC_development/examples/big_spatial"
input_path=$data_path/$(printf "init/init_%s_ns%.3d_ny%.5d_chain%.2d.rds" $modelTypeString $ns $ny $nChains)

for chain in {0..15}; do
    output_path=$PWD/output/$(printf "fmTF_test/TF_%s_ns%.3d_ny%.5d_chain%.2d_sam%.4d_thin%.4d_c%.2d.rds" $modelTypeString $ns $ny $nChains $SAM $THIN $chain)
    echo "$output_path"
    # Remove old output
    rm "$output_path"
    mkdir -p $(dirname $output_path)
    input_chain=$(($chain % 8))  # XXX hack to recalculate old chains as there is no 16 real chains
    export ROCR_VISIBLE_DEVICES=$(($chain % 8))
    python3 ./../../hmsc-hpc/hmsc/examples/run_gibbs_sampler.py --input $input_path --output $output_path --samples $SAM --transient $(($SAM*$THIN)) --thin $THIN --verbose 100 --chain $input_chain --fse 0  --profile $PROFILE &
done
wait

time1=$(date +%s.%N)

echo "Total runtime:" $(echo "$time1 - $time0" | bc) "s"
