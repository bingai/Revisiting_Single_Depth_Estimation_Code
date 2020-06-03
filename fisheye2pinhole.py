import cv2
# assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import sys
# You should replace these 3 lines with the output in calibration step
DIM=(1280, 800)
K = np.array([[3.4404333246422021e+02, 0.0, 6.4384819844899641e+02], [0.0, 3.4476658574610775e+02, 3.9335616307648809e+02], [0.0, 0.0, 1.0]])      
D = np.array([[3.1303260565302769e-02], [1.1532023950485493e-02], [8.0746512028770940e-03], [-4.4565943447174997e-03]])

# def undistort(img_path):
#     img = cv2.imread(img_path)
#     h,w = img.shape[:2]
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     cv2.imshow("undistorted", undistorted_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def main():

    image_dir_fisheye = './data/ValetParking/fisheye_RearCamera/'
    image_names_fisheye = os.listdir(image_dir_fisheye) # list all files in the image folder

    for image_name_fisheye in image_names_fisheye:
        image_name_withPath_fisheye = os.path.join(image_dir_fisheye, image_name_fisheye)
        img_fisheye = cv2.imread(image_name_withPath_fisheye)
        # cv2.imshow("distorted", img_fisheye)
        h, w = img_fisheye.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img_fisheye, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # cv2.imshow("undistorted", undistorted_img)

        image_name_pinhole = image_name_fisheye.replace("fisheye", "pinhole")
        image_dir_pinhole = './data/ValetParking/pinhole_ReaCamera/'
        image_name_withPath_pinhole = os.path.join(image_dir_pinhole, image_name_pinhole)
        cv2.imwrite(image_name_withPath_pinhole, undistorted_img)

        img_fisheye_pinhole_horizontal = np.concatenate((img_fisheye, undistorted_img), axis=1)
        image_name_fisheye_pinhole = image_name_fisheye.replace("fisheye", "fisheye2pinhole")
        image_dir_fisheye2pinhole = './data/ValetParking/fisheye2pinhole/'
        image_name_withPath_pinhole = os.path.join(image_dir_fisheye2pinhole, image_name_fisheye_pinhole)
        cv2.imwrite(image_name_withPath_pinhole, img_fisheye_pinhole_horizontal)



        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
#     for p in sys.argv[1:]:
#         fisheye2pinhole(p)
