import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pdb

imagePath0 = './data/Autohich/image/00000001.png'
imagePath1 = './data/Autohich/image/00000090.png'
image0 = matplotlib.pyplot.imread(imagePath0)
print("image0.size = ", np.shape(image0))
image1 = matplotlib.pyplot.imread(imagePath1)

fig, axs = plt.subplots(1,2)
axs[0].imshow(image1, cmap='gray')
axs[0].set_title("input") 
# axs[0].set_cmap("Greys")

axs[1].imshow(image1, cmap='jet')
axs[1].set_title("depth") 
# axs[1].set_cmap("jet")

plt.show()