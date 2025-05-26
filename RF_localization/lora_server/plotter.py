import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
tag = pd.read_csv('tags.csv')  # Replace with the actual filename
anchors = pd.read_csv('anchors.csv')  # Replace with the actual filename

# Plot X vs Y
plt.plot(tag['X'], tag['Y'], marker='o', linestyle='-', color='blue')
plt.scatter(anchors['X'], anchors['Y'], marker='o', linestyle='-', color='red', label='Anchors')
for _, row in anchors.iterrows():
    plt.text(row['X'], row['Y'], f"A{int(row['Anchor'])}", fontsize=10, ha='right', va='bottom', color='black')


# Add labels and title
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Tag Tracks')
plt.grid(True)

# Show the plot
plt.savefig('plot.png')