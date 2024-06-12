#!/bin/bash -l

#SBATCH -J DPFL
#SBATCH --time=0-9:00:00
#SBATCH --array=2-4
#SBATCH -p gpu
#SBATCH --gres=gpu:teslap100:1    #gtx1080ti
#SBATCH -o outfiles/%A_%a.out

COMMAND='source /storage/homefs/ms23h901/git/DPFL_1/.venv/bin/activate'
echo $COMMAND

COMMAND="python main.py $SLURM_ARRAY_TASK_ID"
echo $COMMAND

$COMMAND

COMMAND='deactivate'
echo $COMMAND
$COMMAND
