# Siamese_Network_Similarity_Learning

This is a private Pytorch practice for the similarity learning with contrastive loss based on the siamese network.

By the way, I use the MNIST dataset as our training data.

Hope this code can help you who find this repo. :)

Paper Reference: <http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf>


## How to use

1. requirements

```
torch=1.10.0
torchvision=0.11.1
```

2. run the python script.

```
python main.py --lr 0.0001 \
               --batch_size 128 \
               --epochs 5 \
               --optimizer adam \
               --normalize \
               --log
```

3. The codes may be updated. To be continued.