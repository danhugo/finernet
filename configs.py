from yacs.config import CfgNode as CN

CFG = CN()

CFG.SYSTEM = CN()
CFG.SYSTEM.PROJECT_PATH = "/home/daoduyhung/hicehehe/finernet"

CFG.SYSTEM.CUDA_VISIBLE_DEVICES = "3"
CFG.SYSTEM.SEED = 42

# WANDB CONFIGS
CFG.WANDB = CN()
CFG.WANDB.ENTITY = "hicehehe"
CFG.WANDB.PROJECT = "finernet"
CFG.WANDB.MODE = "online" # "disabled", "online", "offline"

# DATA
CFG.DATA = CN()
CFG.DATA.DATASET = "cub"
CFG.DATA.FULL_PATH = ""
CFG.DATA.TRAIN_PATH = ""
CFG.DATA.TEST_PATH = ""
CFG.DATA.NUM_CLASSES = 200
CFG.DATA.FAISS_TRAIN_FILE = ""
CFG.DATA.FAISS_TEST_FILE = ""
CFG.DATA.AUG_TRAIN_DATA_PATH = ""



# EXP
CFG.EXP = CN()
CFG.EXP.MODEL = 'vit_b_16'

CFG.EXP.CLASS = CN()
CFG.EXP.CLASS.PATH = ""
CFG.EXP.CLASS.BS = 128
CFG.EXP.CLASS.EPOCH = 100
CFG.EXP.CLASS.FEATURE = 'finernet'
CFG.EXP.CLASS.LR = 1e-3
CFG.EXP.CLASS.ALPHA = 1.0
CFG.EXP.CLASS.FREEZE = False

CFG.EXP.ADVNET = CN()
CFG.EXP.ADVNET.PATH = ""
CFG.EXP.ADVNET.BS = 128
CFG.EXP.ADVNET.EPOCH = 40
CFG.EXP.ADVNET.FEATURE = 'linear'
CFG.EXP.ADVNET.NEG_PRED = 7 # top negative predictions
CFG.EXP.ADVNET.SAMPLE_NEG = 1 # num samples per neg
CFG.EXP.ADVNET.POS = 7
CFG.EXP.ADVNET.LR = 4e-4

configs = CFG
configs.merge_from_file("experiments/resnet50_car.yaml")
configs.freeze()
configs.dump(stream=open('configs.yaml', 'w'))
