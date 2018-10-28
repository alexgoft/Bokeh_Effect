import cv2
import numpy as np

# ============================================================================ #


def get_ref_image(cap, n=10):
    ret_frame = None

    if cap.isOpened():

        initial_frames = []
        for i in range(n):
            ret, frame = cap.read()

            if ret:
                initial_frames.append(frame)
            else:
                break

        ret_frame = np.mean(np.array(initial_frames), axis=0).astype('uint8')

    return ret_frame


def hsv_distance(h, s):
    return np.sqrt(np.square(h) + np.square(s))


def create_binary_map(difference, threshold=20):
    filtered_image = (difference > threshold) * 255

    return filtered_image.astype('uint8')


# ============================================================================ #

def main():
    center = 160, 125

    cap = cv2.VideoCapture('./ball.avi')

    ref_frame = get_ref_image(cap)
    ref_frame_gray = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
    ref_frame_gray_blurred = cv2.GaussianBlur(ref_frame_gray, (31, 31), 0)

    while cap.isOpened() and ref_frame is not None:

        ret, curr_frame = cap.read()
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        curr_frame_gray_blurred = cv2.GaussianBlur(curr_frame_gray, (31, 31), 0)

        curr_frame_blurred = cv2.GaussianBlur(curr_frame, (21, 21), 0)

        if ret:

            diff_image = cv2.absdiff(ref_frame_gray_blurred, curr_frame_gray_blurred)

            binary_map = create_binary_map(diff_image)

            ret, labels = cv2.connectedComponents(binary_map)

            mask = (labels == labels[center[1],center[0]]).astype('uint8')

            if labels[center[1],center[0]]:
                original_ball = cv2.bitwise_and(curr_frame, curr_frame, mask=mask)
                masked_blurred_frame = cv2.bitwise_and(curr_frame_blurred, curr_frame_blurred, mask=1 - mask)

                new_curr_image = original_ball + masked_blurred_frame
            else:
                new_curr_image = curr_frame_blurred
            cv2.circle(new_curr_image,center, 8, (255,0,0))
            cv2.imshow('Result', new_curr_image)
            cv2.imshow('Binary Map', binary_map)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
