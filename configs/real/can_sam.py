_base_ = '../default.py'

expname = 'dvgo_real_can_sam'
basedir = './logs/real'

data = dict(
    datadir='./data/real_can_sam',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)