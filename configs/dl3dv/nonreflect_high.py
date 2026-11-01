_base_ = '../default.py'

expname = 'dvgo_dl3dv_non_reflect_lg'
basedir = './logs/dl3dv'

data = dict(
    datadir='./data/dl3dv_non_reflect',
    dataset_type='tankstemple',
    inverse_y=True,
    white_bkgd=False,
)

fine_train = dict(pg_scale=[1000, 2000, 3000, 4000, 5000, 6000])
fine_model_and_render = dict(num_voxels=256**3)
