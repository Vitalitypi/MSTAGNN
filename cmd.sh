#!/bin/bash
# 搜索最优超参数脚本,随机次数为3
# 主要的超参数有：dim_embedding_periods, dim_embeddings_week, embed_dim, rnn_units, num_layers, kernel, dim_feed_forward
random=("1" "2" "3")
# PEMS03, PEMS04, PEMS07, PEMS08
dataset=('PEMS08')
dim_embed=("4" "8" "10" "18")
use_periods=('0' '1')
use_weekend=('0' '1')
rnn_units=("16" "32" "48" "64" "80")
num_layers=("1" "3" "4" "5")
kernel=("1" "2" "3" "4" "6")
dim_feed_forward=("4" "8" "12" "16" "20")

echo "当前数据集：$dataset"
mkdir ./exps/random/$dataset

for x in "${dim_embed[@]}"
do
    echo "当前参数：dim_embed:$x"
    for rand in "${random[@]}"
    do
        echo "当前随机次数：$rand"
        # 更新conf文件
        sed -i "s/^periods_embedding_dim=.*/periods_embedding_dim= $x/" ./config/${dataset}.conf
        sed -i "s/^weekend_embedding_dim=.*/weekend_embedding_dim= $x/" ./config/${dataset}.conf
        sed -i "s/^embed_dim=.*/embed_dim= $x/" ./config/${dataset}.conf
        sed -i "s/^random=.*/random= True/" ./config/${dataset}.conf
        python main.py --dataset $dataset >> ./exps/random/${dataset}/dim_embed_${x}-random_${rand}.log 2>&1
    done
done
