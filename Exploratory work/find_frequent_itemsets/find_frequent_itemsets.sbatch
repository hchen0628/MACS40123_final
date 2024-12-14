#!/bin/bash

#SBATCH --job-name=find_frequent_itemsets
#SBATCH -o find_frequent_itemsets.out
#SBATCH -e find_frequent_itemsets.err
#SBATCH --partition=caslake
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hchen0628@rcc.uchicago.edu
#SBATCH --account=macs40123

module load python mpich

python3 ./find_frequent_itemsets.py >> find_frequent_itemsets.out
