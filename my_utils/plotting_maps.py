from ipyleaflet import Map, Polyline, Marker, TileLayer, Icon
from ipywidgets import IntSlider, interact
import numpy as np
def plot_trajectory_with_heading(
    latlon_traj,
    headings,
    map_center=None,
    zoom=20,
    width="600px",
    height="650px",
    tile_url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    line_color="yellow"
):
    """
    Plot a trajectory with heading arrows and a pin using ipyleaflet.
    Args:
        latlon_traj: list of (lat, lon) tuples
        headings: list of heading angles in degrees (0° = North)
        map_center: (lat, lon) tuple for map center (default: mean of trajectory)
        zoom: map zoom level (default: 20)
        width, height: map size (default: 900x400 px)
        tile_url: tile server URL (default: Esri World Imagery)
        line_color: color of trajectory line (default: yellow)
    """
    if map_center is None:
        arr = np.array(latlon_traj)
        map_center = (arr[:,0].mean(), arr[:,1].mean())

    m = Map(
        center=map_center,
        zoom=zoom,
        basemap=TileLayer(url=tile_url),
        width=width,
        height=height
    )
    m.layout.width = width
    m.layout.height = height

    traj_line = Polyline(locations=latlon_traj[:1], color=line_color, weight=4, fill=False)
    m.add_layer(traj_line)

    # Arrow icon for heading
    arrow_icon_url = "https://cdn-icons-png.flaticon.com/512/545/545682.png"
    arrow_icon = Icon(
        icon_url=arrow_icon_url,
        icon_size=[30, 30],
        icon_anchor=[20, 20],
    )
    arrow_marker = Marker(
        location=latlon_traj[0],
        icon=arrow_icon,
        rotation_angle=headings[0]-90,
        rotation_origin='center'
    )
    m.add_layer(arrow_marker)

    # Pin icon for current location (default blue)
    pin_marker = Marker(
        location=latlon_traj[0]
    )
    m.add_layer(pin_marker)

    def update(idx):
        traj_line.locations = latlon_traj[:idx+1]
        current_lat, current_lon = latlon_traj[idx]
        heading_deg = headings[idx]
        arrow_marker.location = (current_lat, current_lon)
        arrow_marker.rotation_angle = heading_deg-90
        pin_marker.location = (current_lat, current_lon)

    interact(update, idx=IntSlider(0, 0, len(latlon_traj)-1, 1))
    return m