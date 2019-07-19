from dashdataframe import configure_app
import neuroglancer
import pandas as pd
import dash
import numpy as np


# function to make a neuroglancer link with certain objects selected and pointed at
def render_on_neuroglancer(mesh_ids, dff, id_column='mesh_ids',
                           pos_col = "center_of_mass",
                           center = [96809, 136521, 540]):
    #r= requests.get(blank_state_basil)
    # our base viewer configuration
    base_json = {
    "layers": [
        {
        "source": "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig",
        "type": "image",
        "name": "fafb_v14"
        },
        {
        "source": "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe",
        "type": "image",
        "name": "fafb_v14_clahe",
        "visible": False
        },
        {
        "type": "segmentation",
        "mesh": "precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh",
        "skeletonRendering": {
            "mode2d": "lines_and_points",
            "mode3d": "lines"
        },
        "name": "neuropil-regions-surface"
        },
        {
        "type": "mesh",
        "source": "vtk://https://storage.googleapis.com/neuroglancer-fafb-data/elmr-data/FAFB.surf.vtk.gz",
        "vertexAttributeSources": [],
        "shader": "void main() {\n  emitRGBA(vec4(1.0, 0.0, 0.0, 0.5));\n}\n",
        "name": "neuropil-full-surface",
        "visible": False
        }
    ],
    "navigation": {
        "pose": {
        "position": {
            "voxelSize": [
            4,
            4,
            40
            ],
            "voxelCoordinates": [
            123947.234375,
            73319.125,
            5163.85302734375
            ]
        }
        },
        "zoomFactor": 1210.991144617663
    },
    "perspectiveOrientation": [
        0.05368475615978241,
        -0.002557828789576888,
        -0.02344207465648651,
        -0.9982794523239136
    ],
    "perspectiveZoom": 9221.684888506943,
    "selectedLayer": {
        "layer": "neuropil-regions-surface",
        "visible": True
    },
    "layout": "xy-3d"
    }

    viewer = neuroglancer.Viewer()
    viewer.set_state(base_json)

    with viewer.txn() as s:
        if len(mesh_ids)>0:
            ind = np.where(dff['mesh_id']==mesh_ids[0])[0][0]
            center = np.asarray(dff[pos_col][ind])/[4,4,40] 
            s.navigation.pose.position.voxelCoordinates = center
    
        seg_layer =s.layers['neuropil-regions-surface']
        seg_layer.selectedAlpha = 0.78
        seg_layer.notSelectedAlpha = 0.1
        seg_layer.segments = mesh_ids

    return(neuroglancer.to_url(viewer.state, prefix='https://neuromancer-seung-import.appspot.com/'))

if __name__ == '__main__':
    app = dash.Dash()
    df = pd.read_hdf('fly_regions.h5', 'fly_regions')
    metrics = [c for c in df.columns if (c != 'mesh_id') &(c != 'center_of_mass')]

    configure_app(app, df,
                  link_name='Neuroglancer',
                  create_link=True,
                  link_func=render_on_neuroglancer,
                  id_column='mesh_id',
                  metrics=metrics)
    app.run_server(port=8880)