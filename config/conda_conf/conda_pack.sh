pip install conda-pack
conda pack -n $1 -o $1_packed.tar.gz 
#如果报错后面加上--ignore-missing-files

#mkdir /workspace/work/moniforge3/envs/zhipeng16_cv
#conda create zhipeng16_cv -f