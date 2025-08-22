# コマンドのメモ

nohup rsync -aHAX --partial --append-verify --numeric-ids \
  --info=progress2,stats2 \
  --log-file=/data2/tmurayama/Cosmos/cosmos_copy_$(date +%Y%m%d_%H%M%S).log \
  /data2/tmurayama/Cosmos/{dataset_vript,dataset_ultravideo,dataset_panda70m,checkpoints} \
  /data2/intern01/for_Cosmos/ \
  > /dev/null 2>&1 &
