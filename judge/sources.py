"""Scene lists for each data source, shared by sample.py and scale.py."""
SOURCES = {
    'synthetic_nerf': ['dvgo_lego'],
    'blendedmvs': ['dvgo_Character', 'dvgo_Fountain', 'dvgo_Jade', 'dvgo_Statues'],
    'dtu': ['dvgo_dtu_scan69', 'dvgo_dtu_scan97', 'dvgo_dtu_scan105',
            'dvgo_dtu_scan114', 'dvgo_dtu_scan118'],
    'dl3dv': ['dvgo_dl3dv_non_reflect', 'dvgo_dl3dv_non_reflect2',
              'dvgo_dl3dv_reflect1', 'dvgo_dl3dv_reflect2'],
    'blender': ['dvgo_synth_cat', 'dvgo_synth_lion', 'dvgo_synth_propane'],
    'phone': ['dvgo_real_can_sam_v2'],
}
