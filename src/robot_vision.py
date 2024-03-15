import argparse
from typing import Optional
import apriltag
import numpy as np
import threading
import json
from networktables import NetworkTables
import queue
import signal
import cv2
from april_tags import euler
from roboflow.infer import Roboflow2024

#tag size in centimeters (16.51 centimeters, eg 6.5")
TAG_SIZE = 16.51

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Hyperion 3360 Chargedup 2023 vision application")

    #do we want gui output
    parser.add_argument("-g", "--gui", dest='gui', action='store_true', help="display AR feed from camera with optional AprilTag detection")

    #do we want AprilTag
    parser.add_argument("--apriltag", dest='apriltag', action='store_true', help="enable apriltag detection")

    #do we want AI object detection
    parser.add_argument("--roboflow", dest='roboflow', action='store_true', help="enable ai object detection")

    #print detection results on the console
    parser.add_argument("-v", "--verbose", dest='verbose', action='store_true', help="Display detection results on the console")

    #print detection results on the console
    parser.add_argument("-i", "--ip", dest='rio_ip', type=str, default="10.33.60.2", help="RIO IP address")

    #device from which to acquire
    parser.add_argument("device", type=str, action='store', help="device(s) to capture from. If multiple are given, they must all be of the same resolution", nargs='+')

    #camera parameters as provided by the output of the calibrate_camera.py
    parser.add_argument("-c", "--config", dest='camera_config', nargs='+', default=['camera.json'], action='store', help="json file(s) containing the camera parameters")

    #frame width to acquire
    parser.add_argument("-w", "--width", type=int, default=640, dest='width', action='store', help="capture width from camera")

    #frame height to acquire
    parser.add_argument("--height", type=int, default=480, dest='height', action='store', help="capture height from camera")

    #the game layout of AprilTags in json format
    parser.add_argument("-e", "--environment", dest='environment', default='env.json', action='store', help="json file containing the details of the AprilTags env")

    #record
    parser.add_argument( "--record", type=str, default="/home/data/rec.mp4", help="Record file to specified location",)

    return parser

################################################################################
# April tag subsystme initialization
# Open an parse the game environment JSON
# Open an parse the camera fundamental parameters
################################################################################
def init_april_tag(args):
    tag_info = dict()
    camera_params = []

    #let's load the environment
    with open(args.environment, 'r') as f:
        env_json = json.load(f)
        tag_info = {x['ID']: x for x in env_json['tags']}

    for config in args.camera_config:
        with open(config, 'r') as f:
            cam_json = json.load(f)
            camera_params.append({'params' : [ cam_json[x] for x in ('fx', 'fy', 'cx', 'cy')], 'dist' : cam_json['dist']})

    return tag_info, camera_params

################################################################################
# Communication thread
# Decouple the communication from the rest of the vision system
# Reads a message queue and intepret simple commands
# Push the values in specific sections of network tables
################################################################################
def communication_thread(message_q):
    notified = [False]

    cond = threading.Condition()

    def connectionListener(connected, info):
        print(info, '; Connected=%s' % connected)
        with cond:
            notified[0]=True
            cond.notify()

    NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

    table = None

    while True:

        if not notified[0]:
            with cond:
                cond.wait(0.01)

        if notified[0] and table is None:
            print("Connected!")
            table = NetworkTables.getTable("Vision")

        item = message_q.get()

        if 'command' in item:
            if item['command'] == 'stop':
                break

        else:
            if 'april_tag' in item and table:
                pos,rot = item['april_tag']['position']
                print("Position: {}, rotation: {}".format(pos, rot))
                ids = item['april_tag']['ids']
                table.putNumberArray("AprilTags", [*pos, *rot, *ids])

            if 'note' in item and table:
                print(f"Detection of note at normalized coordinates {item['note']} in the camera fov")
                table.putNumberArray("Note", item['note'])

            if 'detection' in item and table:
                pred = item['detection']
                c, s, x1, y1, x2, y2  = pred
                print( "Detection of {} @ x1: {} y1: {} x2: {} y2: {} with confidence: {}".format(c, x1, y1, x2, y2, s))
                table.putStringArray('detection', [c,str(s),str(x1),str(y1),str(x2),str(y2)])


################################################################################
# Initialize the network table and communication subsystem
# Creates the message queue and the communication thread
# Initialize networktables
################################################################################
def init_network_tables(args):
    msg_q = queue.Queue()
    NetworkTables.initialize(args.rio_ip)
    comm_thread = threading.Thread(target=communication_thread, args=(msg_q, ), daemon=True)

    return comm_thread, msg_q

################################################################################
# Base on the detector results computes the absolute global position and
# associate rotation angles
# TODO: the function assumes the camera is centered within the robot construction
################################################################################
def process_april_tag_detection( camera_params, detector, result, tag_info ):

    # tag36h11. We can allow for some error correcting. No hamming check
    if result.tag_id in tag_info.keys():

        pose, _, _ = detector.detection_pose(result, camera_params['params'] )

        # those angles need to be computed before we manipulate the pose matrix
        angles = euler.rotation_angles(pose[0:3,0:3], 'xyz')

        # get the information of that specific tag
        tag_dict = tag_info.get(result.tag_id)

        if tag_dict:
            # build a quaternion based on the pose rotation and translation
            tag_pose = np.zeros((4,4))
            rot = np.array(tag_dict['pose']['rotation'])
            tag_pose[0:3,0:3] = rot
            T = np.asarray(list(tag_dict['pose']['translation'].values()))
            tag_pose[0:3,3] = T
            tag_pose[3,3] = 1

            estimated_pose = np.array(pose)
            # in order to get the correct distance we need to multiply the TAG_SIZE
            estimated_pose[0][3] *= TAG_SIZE
            estimated_pose[1][3] *= TAG_SIZE
            estimated_pose[2][3] *= TAG_SIZE

            # TODO: investigue if that costly inversion could be replaced by a simple transpose
            tag_relative_camera_pose = np.linalg.inv(estimated_pose)

            global_position = np.matmul(tag_pose, tag_relative_camera_pose)[0:3,3]

            return global_position, angles

    return None

################################################################################
# Generate all the april tag detection results from the detector on the frame
# Then averages the results to increase accuracy
################################################################################
def compute_position( frame, detector, camera_matrix, dist_coeffs, camera_params, tag_info ):
    #work with undistorted image
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)

    #convert to grayscale
    gray = cv2.cvtColor(undistorted , cv2.COLOR_BGR2GRAY)

    #generate detections
    results = detector.detect(gray)

    estimated_poses = []
    # loop over the AprilTag detection results
    for r in results:

        pose = process_april_tag_detection( camera_params, detector, r, tag_info )
        if pose is not None:
            estimated_poses.append(pose)

    if estimated_poses:
        total_pos = np.zeros(3,)
        total_euler = np.zeros(3,)

        # compute average to increase precision and stability
        for position, angles in estimated_poses:
            total_pos += position
            total_euler += angles

        return total_pos / len(estimated_poses), total_euler / len(estimated_poses), results

    return None, None, []

################################################################################
# Draw april tags
# Useful for debugging
################################################################################
def draw_april_tags(frame, tag_detections):
    for result in tag_detections:
         # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = result.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(result.center[0]), int(result.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

################################################################################
# Bounding box coordinates are returned normalized form from the NN
# Simple convenience function
################################################################################
def normalized_to_absolute(x1,y1,x2,y2, w, h):
    x1 *= w
    x2 *= w
    y1 *= h
    y2 *= h

    return (int(x1), int(y1), int(x2), int(y2))

################################################################################
# Draw AI object detection bounding box
# Useful for debugging
################################################################################
def draw_object_detections(frame, predictions):
    for p in predictions:
        _, _, x1, y1, x2, y2  = p
        h, w, _ = frame.shape
        absx1, absy1, absx2, absy2 = normalized_to_absolute(x1, y1, x2, y2, w, h)
        cv2.rectangle( frame, (absx1, absy1), (absx2, absy2), (255,0,0), 5)

################################################################################
# This is the main vision processing loop, it reads frames from camera and
# process april tags and AI detection
################################################################################
def vision_processing(kwargs):
    args = kwargs['args']

    cameras = kwargs['camera']
    camera_params = [
                        {
                            'params': params,
                            'dist': np.asarray(params['dist']),
                            'matrix': np.array([params['params'][0],0, params['params'][2], 0, params['params'][1], params['params'][3], 0, 0, 1]).reshape((3,3))
                        }
                        for params in kwargs['camera_params']
                    ]

    options = apriltag.DetectorOptions( families='tag36h11',
                                        debug=False,
                                        refine_decode=True,
                                        refine_pose=True)
    detector = apriltag.Detector(options)

    tag_info = kwargs['tag_info']
    msg_q = kwargs['comm_msg_q']

    run_ai = kwargs['roboflow']
    model: Optional[Roboflow2024] = None

    if run_ai:
        model = Roboflow2024()

    if args.record:
        video_out = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15, (args.width,args.height))

    fc = -1
    while not kwargs['quit']: # We assume all video captures are open... Either way, cap.read() will fail if the capture isn't open and we can handle the error there
        fc += 1 # frame counter

        april_tags = []
        frame = None # only use the first camera's frame for gui and inference

        for i, cap in enumerate(cameras):
            if args.apriltag:
                params = camera_params[i]
                f, pos, angs, tags = run_april_tags_detection(cap, detector, tag_info, params['params'], params['dist'], params['matrix'])
                if pos is not None:
                    april_tags.append({'pos': pos, 'angles': angs, 'tags': tags})
            else:
                r, f = cap.read()
                if not r:
                    continue

            if frame is None:
                frame = f

        def publish_note(note_coords, msg_q):
                if note_coords:
                    msg_q.put({'note': note_coords})

        if model is not None and (fc % 10) == 0: # Infer once every 10 frames...
            model.infer_async( # USING FIRST CAMERA FOR AI
                frame.copy(),
                lambda note: publish_note(note, msg_q),
                "note"
            )

        if args.apriltag and len(april_tags) > 0:
            detected_tags = []
            pos = np.zeros((len(april_tags), 3))
            angles = np.zeros((len(april_tags), 3))

            for i, detection in enumerate(april_tags):
                detected_tags += [tag_info[tag[1]]['ID'] for tag in detection['tags']]
                pos[i, :] = detection['pos']
                angles[i, :] = detection['angles']

                draw_april_tags(frame, detection['tags'])

            # average position and angle
            pos = np.mean(pos, axis=0)
            angles = np.mean(angles, axis=0)

            print(f"pos: {pos}")
            print(f"angles: {angles}")
            print(f"IDs: {detected_tags}")

            msg_q.put(
                {
                    'april_tag':
                        {
                            'ids': detected_tags,
                            'position': (pos, angles)
                        }
                }
            )

        if args.gui:
            # show the output image after AprilTag detection
            # default to the first camera
            cv2.imshow("Image", frame)
            cv2.waitKey(1)

        if args.record:
            video_out.write(frame)

    if args.record:
        video_out.release()

    if model is not None:
        model.shutdown()

    if args.gui:
        cv2.destroyAllWindows()

    msg_q.put({'command':'stop'})


def run_april_tags_detection(cap, detector, tag_info, camera_params, dist_coeffs, camera_matrix):
    pos = None
    angles = None
    tag_detections = None

    #read a frame
    ret, frame = cap.read()

    #if we have a good frame from the camera
    if ret:
        pos, angles, tag_detections = compute_position( frame, detector, camera_matrix, dist_coeffs, camera_params, tag_info )
        if pos is not None:
            print(f"with tags {tag_detections}:\n\tpos: {pos}\n\tangles: {angles}")

    return frame if ret else None, pos, angles, tag_detections


################################################################################
# setup the camera capture parameter, this version is a simple convenience
# function but a much more elaborate gstreamer pipeline could be used with
# a CSI camera based on nvargus
################################################################################
def setup_capture(dev, w, h):
    cap = cv2.VideoCapture( dev )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


################################################################################
def main():
    kwargs = {}
    kwargs['quit'] = False

    #pretty print numpy
    np.set_printoptions(precision = 3, suppress = True)

    def ctrl_c_handler(signal, frame):
        kwargs['quit'] = True

    signal.signal(signal.SIGINT, ctrl_c_handler)

    parser = build_arg_parser()

    args = parser.parse_args()

    kwargs['args'] = args

    if args.apriltag:
        kwargs['tag_info'],kwargs['camera_params'] = init_april_tag(args)

    kwargs['roboflow'] = args.roboflow

    kwargs['camera'] = [setup_capture(device, args.width, args.height) for device in args.device]
    assert len(kwargs['camera']) == len(kwargs['camera_params']), "Number of camera devices does not match number of camera param files"

    comm_thread, kwargs['comm_msg_q'] = init_network_tables(args)

    comm_thread.start()

    if args.apriltag:
        vision_processing(kwargs)

    comm_thread.join()

    for camera in kwargs['camera'] or []:
        if camera.isOpened():
            camera.release()

#--------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
