_base_ = '../default.py'

expname = 'dvgo_real_can'
basedir = './logs/real'

data = dict(
    datadir='./data/real_can',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)