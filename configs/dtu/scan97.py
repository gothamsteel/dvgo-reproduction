_base_ = '../default.py'

expname = 'dvgo_dtu_scan97'
basedir = './logs/dtu'

data = dict(
    datadir='./data/dtu_tnt/dtu_scan97',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=True,
)
