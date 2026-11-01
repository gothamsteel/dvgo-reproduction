_base_ = '../default.py'

expname = 'dvgo_synth_lion'
basedir = './logs/synth'

data = dict(
    datadir='./data/synth_lion',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)