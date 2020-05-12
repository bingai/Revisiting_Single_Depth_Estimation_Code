import argparse
import numpy as np
import loaddata_demo
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")
import cv2

outputPath = './data/demo/test.png'
print('++++++++++++++++ output info:')
print('++++++ Matplotlib output information:')
output_mat = matplotlib.image.imread(outputPath)
print(' Matplotlib Output info: ', output_mat.shape)  
print(' Matplotlib Output max = {}, min = {} '.format(np.amax(output_mat), np.amin(output_mat)))
matplotlib.pyplot.imshow(output_mat)
matplotlib.pyplot.title("Matplotlib Output")
matplotlib.pyplot.show()

matplotlib_saved_path = './data/demo/matplotlib_saved.png'
matplotlib.image.imsave(matplotlib_saved_path, output_mat)
matplotlib_saved = matplotlib.image.imread(matplotlib_saved_path)
print(' Matplotlib Output_saved info: ', matplotlib_saved.shape)  
print(' Matplotlib Output_saved max = {}, min = {} '.format(np.amax(matplotlib_saved), np.amin(matplotlib_saved)))

print('++++++ CV2 output information:')
output_cv2 = cv2.imread(outputPath)
print(' CV2 Output info: ', output_cv2.shape)  
print(' CV2 Output max = {}, min = {} '.format(np.amax(output_cv2), np.amin(output_cv2)))
output_cv2 = cv2.applyColorMap(output_cv2, cv2.COLORMAP_JET)
cv2.imshow("CV2 Output", output_cv2)
print('-- CV2 Output info: ', output_cv2.shape)  
print('-- CV2 Output max = {}, min = {} '.format(np.amax(output_cv2), np.amin(output_cv2)))
print('------------------------------')
# output_cv2 = cv2.applyColorMap(output_cv2, cv2.COLORMAP_JET)
key = cv2.waitKey(10000)

cv2_saved_path = './data/demo/cv2_saved.png'
cv2.imwrite(cv2_saved_path, output_cv2)
cv2_saved = cv2.imread(cv2_saved_path)
print(' CV2 Output_saved info: ', cv2_saved.shape)  
print(' CV2 Output_saved max = {}, min = {} '.format(np.amax(cv2_saved), np.amin(cv2_saved)))


# cv2.imwrite(cv2_saved_path, output_cv2)
cv2_saved = cv2.imread(cv2_saved_path)
print(' CV2 Output_saved info: ', cv2_saved.shape)  
print(' CV2 Output_saved max = {}, min = {} '.format(np.amax(cv2_saved), np.amin(cv2_saved)))

# inputPath = './data/demo/test.jpg'
# print('----------------- input info:')
# print('------ Matplotlib Input information:')
# input_mat = matplotlib.image.imread(inputPath)
# print(' Matplotlib Input info:: ', input_mat.shape)  
# print(' Matplotlib Input max = {}, min = {}: '.format(np.amax(input_mat), np.amin(input_mat)))
# matplotlib.pyplot.imshow(input_mat)
# matplotlib.pyplot.title("Matplotlib Input")
# matplotlib.pyplot.show()


# print('------ CV2 Input information:')
# input_cv2 = cv2.imread(inputPath)
# print(' CV2 Input info:: ', input_cv2.shape)  
# print(' CV2 Input max = {}, min = {}: '.format(np.amax(input_cv2), np.amin(input_cv2)))
# cv2.imshow("CV2 Input", input_cv2)
# key = cv2.waitKey(10000)