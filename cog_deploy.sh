export MODEL="tiiuae/falcon-40b-instruct"
rm predict.py
envsubst < predict.template.py > predict.py
# cog push r8.im/kunato/rwkv-4-169m-pile
cog push r8.im/kunato/falcon-40b-instruct