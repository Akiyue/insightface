from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.02
config.verbose = 2000
config.dali = False

config.rec = "/kaggle/input/faces-webface-112x112/faces_webface_112x112"
config.num_classes = 10575
config.num_image = 494414
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw']
# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
