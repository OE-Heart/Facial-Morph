import numpy as np
import dlib
import glob
import cv2
from scipy.spatial import Delaunay

input_img_path = "images/input.jpg"
output_img_path = "images/output.jpg"

def face_landmark_detection(image, predictor_model):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    win = dlib.image_window()
    
    img = image
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

def facial_gridding(img_path):
    image = dlib.load_rgb_image(img_path)
    predictor_model = "shape_predictor_81_face_landmarks.dat"
    points = face_landmark_detection(image, predictor_model)

    grid = Delaunay(points)

    return points, grid.simplices

def img_tri_affine(img1, img2, tri1, tri2) :
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)
    
    # Offset points by left top corner of the respective rectangles
    tri1Cropped = []
    tri2Cropped = []
    
    for i in range(0, 3):
        tri1Cropped.append(((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1])))
        tri2Cropped.append(((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1])))
 
    # Crop input image
    img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
 
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
    
    # Apply the Affine Transform just found to the input image
    img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
 
    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)
    img2Cropped = img2Cropped * mask
    
    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]  * ( (1.0, 1.0, 1.0) - mask )  
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]  + img2Cropped

def facial_morph(input_img_path, output_img_path):
    input_img_points, input_img_grid = facial_gridding(input_img_path)
    output_img_points, _ = facial_gridding(output_img_path)

    input_img = cv2.imread(input_img_path)
    output_img = cv2.imread(output_img_path)

    input_img_t = 255 * np.ones(input_img.shape, dtype = input_img.dtype)
    output_img_t = 255 * np.ones(output_img.shape, dtype = output_img.dtype)

    duration = 100
    t = 0
    for i in range(duration):
        for j in range(input_img_grid.shape[0]):
            input_img_grid_points = np.float32([
                input_img_points[input_img_grid[j, 0]],
                input_img_points[input_img_grid[j, 1]],
                input_img_points[input_img_grid[j, 2]]
            ])

            output_img_grid_points = np.float32([
                output_img_points[input_img_grid[j, 0]],
                output_img_points[input_img_grid[j, 1]],
                output_img_points[input_img_grid[j, 2]]
            ])

            mid_grid_points = input_img_grid_points*(1-t)+output_img_grid_points*t

            img_tri_affine(input_img, input_img_t, input_img_grid_points, mid_grid_points)
            img_tri_affine(output_img, output_img_t, output_img_grid_points, mid_grid_points)

        res_image_t = input_img_t[0:651, :, :] * (1-t)/255 + output_img_t[0:651, :, :] * t/255
        
        title = "Facial Morph"
        cv2.namedWindow(title, 0)
        cv2.resizeWindow(title, input_img.shape[1]+20, input_img.shape[0]+20)
        cv2.imshow(title, res_image_t)
        cv2.waitKey(50)
        
        t += 1.0 / duration

if __name__ == "__main__":
    facial_morph(input_img_path, output_img_path)