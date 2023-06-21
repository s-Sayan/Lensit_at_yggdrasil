#!/bin/sh
#SBATCH --partition=shared-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=output.log
#SBATCH --error=error.log

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo ""
echo "***** LAUNCHING *****"
echo `date '+%F %H:%M:%S'`
echo ""


#launch simulations
#conda init bash
which python
srun python epiloge.py

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""