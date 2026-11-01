_base_ = '../default.py'
expname = 'dvgo_dl3dv_non_reflect2'
basedir = './logs/dl3dv'

data = dict(
    datadir='./data/dl3dv_non_reflect2',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=False,
)