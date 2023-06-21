#!/bin/sh
#SBATCH --partition=shared-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
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
pwd

# Retrieve the input number from the command-line argument
number=$1

# Use the input number in your script
echo "The input number is: $number"

output_file=./output/slurm_${number}.out

srun python script.py $number > $output_file

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""
