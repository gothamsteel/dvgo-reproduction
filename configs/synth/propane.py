_base_ = '../default.py'

expname = 'dvgo_synth_propane'
basedir = './logs/synth'

data = dict(
    datadir='./data/synth_propane',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)