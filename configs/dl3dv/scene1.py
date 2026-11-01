_base_ = '../default.py'
expname = 'dvgo_dl3dv_scene1'
basedir = './logs/dl3dv'

data = dict(
    datadir='./data/dl3dv_scene1',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=False,
)