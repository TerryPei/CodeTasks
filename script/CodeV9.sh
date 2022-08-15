#PBS -l ncpus=48
#PBS -l mem=100GB
#PBS -l jobfs=100GB
#PBS -l ngpus=8
#PBS -q gpuvolta
#PBS -P ****
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/****
#PBS -l wd

project_dir='CodeTasks'

module load python3/3.9.2
cd /CodeTasks/

python3 pretrain/CodeV9
