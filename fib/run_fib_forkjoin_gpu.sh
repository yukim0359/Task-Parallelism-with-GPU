#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=00:01:00
#PBS -W group_list=gc64
#PBS -j oe

cd $PBS_O_WORKDIR
./fib_forkjoin_gpu
