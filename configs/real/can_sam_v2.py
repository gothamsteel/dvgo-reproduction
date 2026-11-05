_base_ = '../default.py'

expname = 'dvgo_real_can_sam_v2'
basedir = './logs/real'

data = dict(
    datadir='./data/real_can_sam_v2',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)
