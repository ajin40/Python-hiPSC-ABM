import cv2

def image_to_video():
    """Creates a video out of the png images
    """
    base_path = "C:\\Python27\\MEGA-ARRAY\\88.0\\"

    # path = "C:\\Python27\\Bool Model\\2.0\\*png"
    img_array = []
    for i in range(int(12) + 2):
        img = cv2.imread(base_path + 'network_image' + str(int(i)) + ".png")

        img_array.append(img)

    out = cv2.VideoWriter(base_path + 'network_video.mp4', cv2.VideoWriter_fourcc(*"DIVX"), 0.5, (1500, 1500))

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()


image_to_video()