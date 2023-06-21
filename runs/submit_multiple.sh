#!/bin/bash
conda activate clusterlens
# Loop from 0 to 19 (inclusive)
for number in {0..19}
do
  echo "Submitting job with input number: $number"
  which python
  # Submit the Slurm job with the current input number
  sbatch submit.sh $number
  
  sleep 1  # Sleep for 1 second between job submissions
done
