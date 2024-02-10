from dataclasses import dataclass
import sys
import ctypes
from typing import Generator, Optional
from PIL import Image

#
#
# Get the common support code for the GigE-V Framework for Linux
# (Change this if directory structure is changed).
import os
from utils import ipAddr_to_string, to_PIL
sys.path.append(os.path.dirname(__file__) + "/gigev_common")

import pygigev  # includeded in ../gigev_common, DO NOT install from pip
from pygigev import GevPixelFormats as GPF

@dataclass
class BufferParams:
    width: int
    height: int
    pixel_bitsize: int
    payload_size: int
    pixel_format: any
    _bufsize: Optional[int] = None

    @property
    def unpacked_size(self):
        return (self.width, self.height) * self.pixel_bitsize

    @property
    def size(self):
        if self._bufsize is not None:
            return self._bufsize

        bufsize = self.payload_size

        unpacked_size = self.unpacked_size
        if unpacked_size > self.payload_size:
            bufsize = unpacked_size

        return bufsize

    def freeze(self):
        self._bufsize = self.size

class CameraInfo:
    @staticmethod
    def list_cameras(max_n: int = 4):
        numFound = (ctypes.c_uint32)(0)
        camera_info = (pygigev.GEV_CAMERA_INFO * max_n)()

        # Get the camera list
        status = pygigev.GevGetCameraList(camera_info, max_n, ctypes.byref(numFound))
        if status != 0:
            raise Exception(f"Error {status} getting camera list")

        if numFound.value == 0:
            return []


        print(f"{numFound.value} cameras found")
        return camera_info[:numFound.value]

    @staticmethod
    def parse_ips(cameras):
        ips = []
        for cam in cameras:
            ips.append(ipAddr_to_string(cam.ipAddr))

        return ips

    @staticmethod
    @property
    def dimen_names():
        if sys.version_info > (3, 0):
            width_name = b"Width"
            height_name = b"Height"
        else:
            width_name = "Width"
            height_name = "Height"

        return width_name, height_name

class Camera:
    def __init__(self, ip: Optional[str] = None, index: Optional[str] = None):
        available_cameras = CameraInfo.list_cameras()
        ips = CameraInfo.parse_ips(available_cameras)

        if ip == None:
            index = index or 0
            self.ip = ips[index]
            self._camera_info = available_cameras[index]
        else:
            if ip not in ips:
                raise Exception(f"Could not find camera at specified ip. Available ips are: {ips}")
            self.ip = ip
            self._camera_info = available_cameras[ips.index(ip)]

        self._handle = None
        self._buffer_count = None
        self._buf_params: Optional[BufferParams] = None

    def open(self):
        print(f"Opening camera {self.ip}")

        handle = (ctypes.c_void_p)()
        status = pygigev.GevOpenCamera(self._camera_info, pygigev.GevExclusiveMode, ctypes.byref(handle))
        # TODO check status

        self._handle = handle
        self._buf_params = self._get_payload_params()

    def setup(self, buffer_count: int = 10):
        """
        TODO: Check if we need to call the setup function before each read
        """
        self._buffer_count = buffer_count
        params = self._buf_params

        if params is None:
            raise Exception("Please open the camera first with Camera.open() !")

        # Allocate buffers to store images in (2 here).
        # (Handle cases where image is larger than payload due to pixel unpacking)
        buffer_addresses = ((ctypes.c_void_p) * buffer_count)()

        for bufIndex in range(buffer_count):
            temp = ((ctypes.c_char) * params.size)()
            buffer_addresses[bufIndex] = ctypes.cast(temp, ctypes.c_void_p)

        # Initialize a transfer (Asynchronous cycling)
        print("Init transfer :")
        status = pygigev.GevInitializeTransfer(
            self._handle, pygigev.Asynchronous, params.payload_size, buffer_count, buffer_addresses
        )

        # TODO handle status

    def read(self) -> Generator[Image.Image, None, None]:
        params = self._buf_params
        if params is None:
            raise Exception("Please setup the camera first with Camera.setup() !")

        n_img = self._buffer_count

        # Grab images to fill the buffers
        status = pygigev.GevStartTransfer(self._handle, n_img)

        # Read the images out
        gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()

        for imgIndex in range(n_img):
            tmout = (ctypes.c_uint32)(1000)
            status = pygigev.GevWaitForNextFrame(self._handle, ctypes.byref(gevbufPtr), tmout.value)

            # Check for dropped frame
            # Check img data status
            gevbuf = gevbufPtr.contents
            if status != 0 or gevbuf.status != 0:
                # print("Img # ", imgIndex, " has status = ", gevbuf.status)
                continue

            print(f"Img {imgIndex} : id = {gevbuf.id} w = {gevbuf.w} h = {gevbuf.h} address = {hex(gevbuf.address)}")

            # Make a PIL image out of this frame (assume 8 bit Mono (also works for 8bit Bayer before decoding))
            im_size = (gevbuf.w, gevbuf.h)
            im_addr = ctypes.cast(gevbuf.address, ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size))

            yield to_PIL(params.pixel_format, im_size, im_addr)

    def release(self):
        # Free the transfer
        print("Free transfer :")
        status = pygigev.GevFreeTransfer(self._handle)

        # Close the camera
        print("Close camera :")
        status = pygigev.GevCloseCamera(ctypes.byref(self._handle))

        # TODO handle status


    def _get_payload_params(self) -> BufferParams:
        if self._handle is None:
            raise Exception("Camera was not opened")

        # Get the payload parameters
        print("Getting payload information for {self.ip}:")

        payload_size = (ctypes.c_uint64)()
        pixel_format = (ctypes.c_uint32)()

        status = pygigev.GevGetPayloadParameters(self._handle, ctypes.byref(payload_size), ctypes.byref(pixel_format))
        pixel_format_unpacked = pygigev.GevGetUnpackedPixelType(pixel_format)

        # print(
        #     "status :",
        #     status,
        #     " payload_size : ",
        #     payload_size.value,
        #     " pixel_format = ",
        #     hex(pixel_format.value),
        #     " pixel_format_unpacked = ",
        #     hex(pixel_format_unpacked),
        # )

        # Get the Width and Height (extra information)
        feature_strlen = (ctypes.c_int)(pygigev.MAX_GEVSTRING_LENGTH)
        unused = (ctypes.c_int)(0)

        width_str = ((ctypes.c_char) * feature_strlen.value)()
        height_str = ((ctypes.c_char) * feature_strlen.value)()

        width_name, height_name = CameraInfo.dimen_names

        status = pygigev.GevGetFeatureValueAsString(self._handle, width_name, unused, feature_strlen, width_str)
        status = pygigev.GevGetFeatureValueAsString(self._handle, height_name, ctypes.byref(unused), feature_strlen, height_str)

        # TODO handle the status if it fails
        print("status :", status, " Width : ", width_str.value, " Height = ", height_str.value)

        width = width_str.value
        height = height_str.value

        params = BufferParams(
                int(width),
                int(height),
                pygigev.GevGetPixelSizeInBytes(pixel_format_unpacked),
                payload_size.value,
                pixel_format
            )
        params.freeze()

        print(f"Using bufsize = {params.size}")

        return params


if __name__ == "__main__":
    # Initialize the API
    pygigev.GevApiInitialize()

    camera = Camera(index=0)
    camera.open()
    camera.setup(buffer_count=1)


    frame = camera.read()

    print(frame)

    camera.release()

