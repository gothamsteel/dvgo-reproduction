_base_ = '../default.py'
expname = 'dvgo_dl3dv_non_reflect'
basedir = './logs/dl3dv'

data = dict(
    datadir='./data/dl3dv_non_reflect',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=False,
)