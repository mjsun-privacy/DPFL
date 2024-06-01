#!/bin/bash -l

#SBATCH -J dspoddfl
#SBATCH --time=0-6:00:00
#SBATCH --array=0-107
#SBATCH -p epyc2
#SBATCH -o outfiles/%A_%a.out

COMMAND='source /storage/homefs/ms23h901/git/DPFL_1/.venv/bin/activate'
echo $COMMAND

COMMAND="python main.py $SLURM_ARRAY_TASK_ID"
echo $COMMAND

$COMMAND

COMMAND='deactivate'
echo $COMMAND
$COMMAND

