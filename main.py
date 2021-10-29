import numpy as np
import dlib
import glob
import cv2
from scipy.spatial import Delaunay

input_pic_path = "images/input.jpg"
output_pic_path = "images/output.jpg"

def face_landmark_detection(image, predictor_model):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    win = dlib.image_window()

    img = dlib.load_rgb_image(image)

    img_height = img.shape[0]
    img_width = img.shape[1]
    print("img_height = ", img_height)
    print("img_width = ", img_width)

    win.clear_overlay()
    win.set_image(img)

    dets = detector(img, 0)
    points = np.zeros((81+8, 2))

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        for num in range(shape.num_parts):
            # cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (255, 0, 0), -1)
            points[num][0] = int(shape.parts()[num].x)
            points[num][1] = int(shape.parts()[num].y)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
    win.add_overlay(dets)

    dlib.hit_enter_to_continue()

    points[81] = [2, 2]
    points[82] = [img_width-2, 2]
    points[83] = [2, img_height-2]
    points[84] = [img_width-2, img_height-2]
    points[85] = [int(img_width/2), 2]
    points[86] = [img_width-2, int(img_height/2)]
    points[87] = [2, int(img_height/2)]
    points[88] = [int(img_width/2), img_height-2]

    return points

def facial_gridding(image_path):
    predictor_model = "shape_predictor_81_face_landmarks.dat"
    points = face_landmark_detection(image_path, predictor_model)

def facial_morph(input_pic_path, output_pic_path):
    facial_gridding(input_pic_path)
    facial_gridding(output_pic_path)

if __name__ == "__main__":
    facial_morph(input_pic_path, output_pic_path)