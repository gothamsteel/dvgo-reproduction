_base_ = '../default.py'

expname = 'dvgo_synth_cat_colmap'
basedir = './logs/synth'

data = dict(
    datadir='./data/synth_cat_colmap',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)