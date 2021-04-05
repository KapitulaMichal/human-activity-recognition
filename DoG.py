import cv2
import numpy as np


def DoG(input_image, imgScale):
    #fn = input("Enter image file name and path: ")
    fn = input_image
    fn_no_ext = fn.split('.')[0]
    outputFile = fn_no_ext + 'DoG.png'
    # read the input file
    img = cv2.imread(str(fn))
    #img = cv2.resize(img,(int(img.shape[1]*imgScale), int(img.shape[1]*imgScale)))

    # run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur5 = cv2.GaussianBlur(img, (9, 9), 0)
    blur3 = cv2.GaussianBlur(img, (7, 7), 0)

    # write the results of the previous step to new files
    cv2.imwrite(fn_no_ext + '3x3.png', blur3)
    cv2.imwrite(fn_no_ext + '5x5.png', blur5)

    DoGim = blur5 - blur3
    cv2.imwrite(outputFile, DoGim)


def compareImages():
    input1 = input('enter the first image to be compared: ')
    input2 = input('enter the second image to be compared: ')
    outFile = input('enter the filename of the desired output: ')

    in1 = cv2.imread(input1)
    in2 = cv2.imread(input2)

    output1 = in2 * -1 * in1

    cv2.imwrite(outFile + '.jpg', output1)

'''
print("Welcome to the Difference of Gaussian Image creation utility.\n")
actionSelection = input(
    "If you would like to compare two images type \"Compare\", if you want to calculate a new set Difference of Gaussian image type \"New\": \n").lower()

if actionSelection == "new":
    DoG()
elif actionSelection == "compare":
    compareImages()
else:
    print("Not a valid selection.")
'''

DoG('frame2.png', 0.5)