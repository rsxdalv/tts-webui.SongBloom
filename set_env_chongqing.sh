export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$(pwd)/third_party"
# export TRANSFORMERS_CACHE="$(pwd)/third_party/hub"
# source /opt/conda/bin/activate 
export TORCH_HOME="$(pwd)/third_party/torch"
export NCCL_HOME=/usr/local/tccl


export NLTK_DATA=$(pwd)/SongBloom/datasets/cn_zh_g2p/nltk_data
export PYTHONPATH="$(pwd)":$PYTHONPATH
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1