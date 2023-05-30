FPS = [0.127,0.427, 1.49, 6.1, 18.5, 29.5]
Resolution = ["1920x1080", "960x540", "480x320", "240x160","120x80","60x40"]

import matplotlib.pyplot as plt

plt.bar(Resolution,FPS,color='g')
plt.title("Photorealistic view rendering speed",fontweight="bold")
plt.xlabel("Rendering resolution",fontweight="bold")
plt.ylabel("FPS",fontweight="bold")
plt.savefig("humannerffps.png")
plt.show()



romp_FPS = [12.8, 28.5] 
romp_resolution = ["1920x1080", "960x540"]


plt.bar(romp_resolution,romp_FPS,color='r')
plt.title("3D mesh rendeirng speed",fontweight="bold")
plt.xlabel("Rendering resolution",fontweight="bold")
plt.ylabel("FPS")
plt.savefig("romp.png")

plt.show()


segment_FPS = [55.5,333.3]
segment_resolution = ["1920x1080", "960x540"]
plt.bar(romp_resolution,romp_FPS,color='b')
plt.title("Segmentation rendering speed",fontweight="bold")
plt.xlabel("Rendering resolution",fontweight="bold")
plt.ylabel("FPS")
plt.savefig("segment.png")
plt.show()

