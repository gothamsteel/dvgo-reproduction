_base_ = '../default.py'

expname = 'dvgo_synth_cat'
basedir = './logs/synth'

data = dict(
    datadir='./data/synth_cat',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)