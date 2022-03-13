import cv2
import multiprocessing
import numpy as np
from multiprocessing import Pool

# write the final side by side image for flight travel
def create_flight_direction_output(video_name, original_image, depth_image, output_path, flight_path):
    # take the depth image and draw the flight_path arrow onto it
    depth_with_arrow_path = cv2.arrowedLine(depth_image, (788, 246), (int(flight_path[0]), int(flight_path[1])), (255,255,255), 3)
    # resize the depth image to be the same as the original video frame (1920x1080)
    resized_final_depth_flight = cv2.resize(depth_with_arrow_path, (1920, 1080), interpolation = cv2.INTER_AREA)
    # create comparison images -- original on top, depth with flight path on the bottom
    stacked_final_result = np.concatenate((original_image, resized_final_depth_flight), axis=0)
    cv2.imwrite(output_path, stacked_final_result)

    return stacked_final_result

# take deepest path and calculate the endpoint of the arrow to be drawn
def calculate_flight_direction(deepest_zone_coordinates):
    top_left = deepest_zone_coordinates[0]
    bottom_right = deepest_zone_coordinates[1]
    # we want to go to the center of the zone
    #   this could be tweaked later to be more precise, not sure if we want to be that precise though
    #   it might cause jerking in the flight so we will keep it to the zoning
    #  **we probably could make something more sophisticated leveraging zones and then smoothing of the depth intensities
    #       i think that goes past what this project is meant to do though
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    return center_x, center_y

def calculate_deepest_point(image):
    # we have a 1576x492 image 
    # we are going to break it down into a grid of 12 rows and 8 columns
    # we will use the idea of a kernel of 197x41 (width x height) to look at each of our zones
    #   we will find out the average zone depth per region
    # we will try to find a zone closest to the center of the image before deciding a far-away zone is the deepest
    # we then take the center point of the image
    height = 492
    width = 1576
    kernel_height = 41 # 41
    kernel_width = 197 # 197
    kernel_rows = int(height/kernel_height) # 12
    kernel_columns = int(width/kernel_width) # 8

    # get the grayscale of the image
    # most deep == most dark == closest to 0
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # these will be for tracking the deepest point as we go through the image
    deepest_zone_average = 255
    deepest_zone_row = 0
    deepest_zone_column = 0
    for r in range(kernel_rows - 2): # (0-9) -- removing the bottom two rows since the heatmap is skewed
        for c in range(kernel_columns): # (0-7)
            starting_row = r*kernel_height
            starting_column = c*kernel_width
            running_total = 0
            for k_r in range(kernel_height):
                for k_c in range(kernel_width):
                    running_total += image_grayscale[starting_row + k_r][starting_column + k_c]
            kernel_average = running_total / (kernel_height*kernel_width)
            # we need to favor the center of the picture much more than the perimetet
            #   we will take the zones from a rectangle in the center and decrease them by 50%
            #   but other zones will stay as is
            if r > 1 and r < 8 and c > 1 and c < 6:
                kernel_average *= .5
            if (kernel_average < deepest_zone_average):
                deepest_zone_average = kernel_average
                deepest_zone_row = r
                deepest_zone_column = c
    
    # find the spot in the matrix that is the deepest zone
    top_left = [deepest_zone_column*kernel_width, deepest_zone_row*kernel_height]
    bottom_right = [(deepest_zone_column+1)*kernel_width, (deepest_zone_row+1)*kernel_height]

    return calculate_flight_direction([top_left, bottom_right])

def process_depth_calculations(video_name):
    datasets_folder = "datasets/"
    video_file = datasets_folder + video_name + ".mp4"
    # reading the video
    capture = cv2.VideoCapture(video_file)
    frame_count = 0
    final_output_frames = []
    while capture.isOpened():
        # video frame -- original image
        ret, original_image = capture.read()
        depth_image = cv2.imread(datasets_folder + video_name + "/" + str(frame_count) + ".png")

        # if there are no more depth images then we stop
        if type(depth_image) == type(None):
            break

        # build the output path
        output_path = datasets_folder + video_name + "/final_comparison_" + str(frame_count) + ".png"

        # deepest point
        deepest_x, deepest_y = calculate_deepest_point(depth_image)

        # create final output
        result = create_flight_direction_output(video_name, original_image, depth_image, output_path, [deepest_x, deepest_y])
        final_output_frames.append(result)
        frame_count += 1
    # create a video of final output files
    final_video_name = datasets_folder + video_name + "/final_comparison_" + video_name + ".mp4"
    final_video = cv2.VideoWriter(final_video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1920, 2160))
    for i in range(len(final_output_frames)):
        final_video.write(final_output_frames[i])
    final_video.release()


if __name__ == "__main__":
    input_videos = ["good_path", "floor", "right_wall"]
    
    # multiprocessing of the videos - calulating the Deepest Zones and creating the output images
    pool = multiprocessing.Pool(3)
    zip(pool.map(process_depth_calculations, input_videos))