import numpy as np
import dlib
import cv2
from scipy.interpolate import CubicSpline
import time


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

class RemoveTeeth:
    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(predictor_path)

    def remove(self):
        img_file = './Smile.jpg'
        img = cv2.imread(img_file)
        img = image_resize(img,width=300)
        dets = self.detector(img, 1)
        height = np.size(img,0)
        width = np.size(img,1)
        x_mouth = []
        y_mouth = []
        # del x, y
        x = []
        y = []
        l = []
        c_img = np.zeros((height,width,4))
        for k, d in enumerate(dets):
            # Detect face key points in box d
            shape = self.predictor(img, d)
            for j in range(68):
                x.append(shape.part(j).x)
                y.append(shape.part(j).y)
            for i in range(20):
                x_mouth.append(shape.part(i+48).x)
                y_mouth.append(shape.part(i+48).y)


        for i in range(17):
            l.append((x[i],y[i]))
        #Key points for mouth's inner edge
        xm1 = x_mouth[12:17]
        ym1 = y_mouth[12:17]
        xm2 = [x_mouth[12],x_mouth[-1], x_mouth[-2],x_mouth[-3],x_mouth[16]]
        ym2 = [y_mouth[12],y_mouth[-1], y_mouth[-2],y_mouth[-3],y_mouth[16]]

        #Interpolation
        cs1 = CubicSpline (xm1, ym1)
        cs2 = CubicSpline (xm2, ym2)

        cs4 = CubicSpline(x[8:15], y[8:15])
        cs4_int = CubicSpline.integrate(cs4,x[8],x[16])
        cs3 = CubicSpline(y[0:8], x[0:8])
        cs3_int = CubicSpline.integrate(cs3,y[0],y[8])
        # c_img = cv2.cvtColor (img, cv2.COLOR_BGR2BGRA)
        # removing Teeth
        for i in range (height) :
            for j in range (width) :
                if cs2 (j) >= i >= cs1 (j) :
                    img [i, j, :] = (0, 0, 0)

        cv2.imwrite ('Teethless.png', img)



if __name__ == "__main__":
    time1 = time.time()
    rt = RemoveTeeth()
    time2 = time.time()
    rt.remove()
    time3 = time.time()
    rt.remove()
    time4 = time.time()

    print("initial -> ", (time2 - time1))
    print("first -> ", (time3 - time2))
    print("second -> ", (time4 - time3))
