#!/bin/bash
#SBATCH --job-name=plant_disease_diagnosis
#SBATCH --account=PAA0023
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=40 --gpus-per-node=1
#SBATCH --output=/users/PAA0023/dong760/plant_leaves_diagnosis/outputs/InceptionV3_model(256x256)_16_0.2ValSplit_03-21-2021
#SBATCH --mail-type=END # BEGIN,END,NONE,FAIL,ALL
#SBATCH --mail-user=dong.760@osu.edu

# If you want interactive allocation: $ salloc -A PAA -t 00:20:00  -N 1 --ntasks-per-node=3 -p debug-40core --gpus-per-node=G

# ENV setup
echo 'environemnt set up'
source ./miniconda3/bin/activate
# module load cuda/10.1.168 # ==> No longer working for tensorflow==2.4.0
module load cuda/11.0.3 
export PYTHONNOUSERSITE=true
conda activate tf_latest

# Other info for writing batch script
# scontrol show node $SLURMD_NODENAME
#SBATCH --partition=gpuserial-48core # $sinfo, $squeue, $scontrol show partition, refers to https://slurm.schedmd.com/quickstart.html, https://www.osc.edu/supercomputing/knowledge-base/slurm_migration/how_to_prepare_slurm_job_scripts
#SBATCH --output=/users/PAA0023/dong760/plant_leaves_diagnosis/outputs/MobileNetV3Small_model_BatchSize_32_0.2ValSplit_19-12-2020 #  baseline_NASNet_test
# #SBATCH --output=/users/PAA0023/dong760/plant_leaves_diagnosis/outputs/plant_gpu_48cores_vgg19_sparse_categorical
# #SBATCH --output=/users/PAA0023/dong760/plant_leaves_diagnosis/outputs/test-00

# $ scontrol show partition
# PartitionName=gpuserial-48core
#    AllowGroups=ALL DenyAccounts=pcon0060,pcon0003,pcon0014,pcon0015,pcon0016,pcon0008,pcon0010,pcon0009,pcon0020,pcon0022,pcon0023,pcon0024,pcon0025,pcon0026,pcon0040,pcon0041,pcon0080,pcon0100,pcon0101,pcon0120,pcon0140,pcon0160,pcon0180,pcon0200,pcon0009,pcon0020,pcon0022,pcon0023,pcon0024,pcon0025,pcon0026,pcon0040,pcon0041,pcon0080,pcon0100,pcon0101,pcon0120,pcon0140,pcon0160,pcon0180,pcon0200,pcon0009,pcon0020,pcon0022,pcon0023,pcon0024,pcon0025,pcon0026,pcon0040,pcon0041,pcon0080,pcon0100,pcon0101,pcon0120,pcon0140,pcon0160,pcon0180,pcon0200 AllowQos=ALL
#    AllocNodes=ALL Default=NO QoS=pitzer-gpuserial-partition
#    DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
#    MaxNodes=1 MaxTime=7-00:00:00 MinNodes=0 LLN=NO MaxCPUsPerNode=48
#    Nodes=p03[01-19]
#    PriorityJobFactor=2000 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
#    OverTimeLimit=NONE PreemptMode=OFF
#    State=UP TotalCPUs=912 TotalNodes=19 SelectTypeParameters=NONE
#    JobDefaults=(null)
#    DefMemPerCPU=7744 MaxMemPerCPU=7744


# Run the program
echo 'Running the batch script'
python plant_leaves_diagnosis/baseline_InceptionV3.py # baseline_backup.py baseline_InceptionV3.py baseline_ResNet.py baseline_debug.py baseline_NASNet.py

echo 
qstat -u dong760 
# squeue -u dong760
echo 'The date when running current script is :'
date

# Other things
# Reference: 
# - Job Scripts, https://www.osc.edu/supercomputing/batch-processing-at-osc/job-scripts
# - Batch-Related Command Summary, https://www.osc.edu/supercomputing/batch-processing-at-osc/batch-related-command-summary
# - How to Submit, Monitor and Manage Jobs, https://www.osc.edu/supercomputing/knowledge-base/slurm_migration/how_to_monitor_and_manage_jobs
# - Job Environment Variables, https://www.osc.edu/supercomputing/knowledge-base/slurm_migration/how_to_prepare_slurm_job_scripts
# - JOb submission, https://www.osc.edu/supercomputing/batch-processing-at-osc/job-submission
# - Manage account, project balance with OSCusage, https://www.osc.edu/resources/getting_started/osc_custom_commands/oscusage
# - Other command related to slurm scheduler, https://slurm.schedmd.com/quickstart.html

