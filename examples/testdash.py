from dashdataframe import configure_app
import neuroglancer
import pandas as pd
import dash
import numpy as np
import os

# function to make a neuroglancer link with certain objects selected and pointed at
def render_on_neuroglancer(
    mesh_ids,
    dff,
    id_column="mesh_ids",
    pos_col="center_of_mass",
    center=[96809, 136521, 540],
):
    # r= requests.get(blank_state_basil)
    # our base viewer configuration
    base_json = {
        "layers": [
            {
                "source": "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig",
                "type": "image",
                "name": "fafb_v14",
            },
            {
                "source": "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe",
                "type": "image",
                "name": "fafb_v14_clahe",
                "visible": False,
            },
            {
                "type": "segmentation",
                "mesh": "precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh",
                "selectedAlpha": 0.78,
                "notSelectedAlpha": 0.1,
                "segments": ["22", "59"],
                "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
                "name": "neuropil-regions-surface",
            },
        ],
        "navigation": {
            "pose": {
                "position": {
                    "voxelSize": [4, 4, 40],
                    "voxelCoordinates": [124416, 67072, 3531.5],
                }
            },
            "zoomFactor": 4,
        },
        "perspectiveOrientation": [
            0.0033587864600121975,
            -0.005099608097225428,
            -0.0007366925710812211,
            0.9999811053276062,
        ],
        "perspectiveZoom": 10761.580012248845,
        "layout": "3d",
    }

    viewer = neuroglancer.Viewer()
    viewer.set_state(base_json)

    with viewer.txn() as s:
        if len(mesh_ids) > 0:
            ind = np.where(dff["mesh_id"] == mesh_ids[0])[0][0]
            center = np.asarray(dff[pos_col][ind]) / [4, 4, 40]
            # s.navigation.pose.position.voxelCoordinates = center

        seg_layer = s.layers["neuropil-regions-surface"]
        seg_layer.selectedAlpha = 0.78
        seg_layer.notSelectedAlpha = 0.1
        seg_layer.segments = mesh_ids

    return neuroglancer.to_url(
        viewer.state, prefix="https://neuromancer-seung-import.appspot.com/"
    )


if __name__ == "__main__":
    app = dash.Dash()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_hdf(os.path.join(dir_path, "fly_regions.h5"), "fly_regions")
    df.set_index("mesh_id")
    configure_app(app, df, link_name="Neuroglancer", link_func=render_on_neuroglancer)
    app.run_server(port=8880)
