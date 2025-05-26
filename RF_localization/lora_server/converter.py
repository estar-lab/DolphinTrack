from gmplot import gmplot
import math
import matplotlib.pyplot as plt

# Convert geodetic coordinates (latitude, longitude, height) to Cartesian coordinates (x, y, z).
# Flat earth approximation is used for simplicity.
def geodetic_to_cartesian(lat_deg, long_deg, lat0_deg, long0_deg):
    lat = math.radians(lat_deg)
    long = math.radians(long_deg)

    lat0 = math.radians(lat0_deg)
    long0 = math.radians(long0_deg)

    # Earth's radius in meters (WGS84)
    # This is a simplified model, assuming a spherical Earth.
    R = 6378137

    # Compute deltas
    dx = (long - long0) * math.cos(lat0) * R
    dy = (lat - lat0) * R

    return dx, dy

# Meters, helpful for generating final postions
anchor_heights = [
    1.47,  # anchor 1 height
    1.8,  # anchor 2 height
    1.05,  # anchor 3 height
    0.55,  # anchor 4 height
    1.8,  # anchor 5 height
]

# Anchor 1 is always defined as the origin (0, 0)
# z=0 is defined as the ground
anchors = [21.27190593517179, -157.77298863113072, # anchor 1
           21.271873334052845, -157.77318192333698, # anchor 2
           21.272084538994143, -157.77333346814441, # anchor 3
           21.272219396588447, -157.77321265374803, # anchor 4
           21.272059027507208, -157.77294370413026, # anchor 5
]
lats = anchors[0::2]
lons = anchors[1::2]

gmap = gmplot.GoogleMapPlotter(anchors[0], anchors[1], 20, maptype='satellite')
gmap.scatter(lats, lons, '#FF0000', size=0.5, marker=False)
gmap.draw("map.html")

# Convert anchor positions to Cartesian coordinates
anchor_positions = {}
for i in range(5):
    lat, lon = anchors[i*2], anchors[i*2 + 1]
    x, y = geodetic_to_cartesian(lat, lon, lats[0], lons[0])
    anchor_positions[i + 1] = [x, y, anchor_heights[i]]

# Print the anchor positions
for anchor_id, position in anchor_positions.items():
    print(f"Anchor {anchor_id}: {position}")

# Plot the anchors
plt.scatter([pos[0] for pos in anchor_positions.values()],
            [pos[1] for pos in anchor_positions.values()],
            color='red', label='Anchors')
for i, (x, y, z) in anchor_positions.items():
    plt.text(x, y , str(i), fontsize=9, ha='right', va='bottom', color='black')  # slight offset to avoid overlapping
plt.title('Anchor Positions')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.legend()
plt.savefig("anchors.png")

# Print the anchor positions in a CSV
with open('anchors.csv', 'w') as f:
    f.write('Anchor,X,Y,Z\n')
    for anchor_id, position in anchor_positions.items():
        f.write(f"{anchor_id},{position[0]},{position[1]},{position[2]}\n")