export MODEL="RWKV/rwkv-4-169m-pile"
rm predict.py
envsubst < predict.template.py > predict.py
cog push r8.im/kunato/rwkv-4-169m-pile