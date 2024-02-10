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

import cv2
import numpy as np
from utils import ipAddr_to_string
sys.path.append(os.path.dirname(__file__) + "/gigev_common")

import pygigev  # includeded in ../gigev_common, DO NOT install from pip
from pygigev import GevPixelFormats as GPF


def to_PIL(pixel_format, im_size, im_addr):
    # Read from buffer and create an image.
    # | camera buffer | Pillow image |
    # | ------------- | ------------ |
    # | RGB8          | RGB8         |
    # | BGR8          | RGB8         |
    # | RGBA8         | RGBA8        |
    # | BGRA8         | RGBA8        |
    # | UYVY          | YCbCr        |
    # | YUYV          | YCbCr        |
    # | Mono8         | Mono8        |
    # | 8-bit bayer   | Mono8(bayer) |
    #
    if pixel_format.value == GPF.fmtRGB8Packed.value:
        print("RGB8 mode")
        im = Image.frombuffer("RGB", im_size, im_addr.contents, "raw", "RGB", 0, 1)

    elif pixel_format.value == GPF.fmtBGR8Packed.value:
        print("BGR8 mode")
        im = Image.frombuffer("RGB", im_size, im_addr.contents, "raw", "BGR", 0, 1)

    elif pixel_format.value == GPF.fmtRGBA8Packed.value:
        print("RGBA8 mode")
        im = Image.frombuffer(
            "RGBA", im_size, im_addr.contents, "raw", "RGBA", 0, 1
        )

    elif pixel_format.value == GPF.fmtBGRA8Packed.value:
        # PIL v9.4.0 doesn't support BGRA mode yet, so it's trickier than RGBA.
        print("BGRA8 mode")

        # Read as RGBA, but it's actually BGRA
        im = Image.frombuffer(
            "RGBA", im_size, im_addr.contents, "raw", "RGBA", 0, 1
        )

        # Convert BGRA to RGBA
        b, g, r, a = im.split()
        im = Image.merge("RGBA", (r, g, b, a))

    elif pixel_format.value == GPF.fmtYUV422packed.value:
        print("YUV422_8_UYVY mode")

        im = YUV422_UYVY_Decoder(im_size, im_addr.contents)

        # [Optional] Convert the image from YCbCr to RGB
        # im = im.convert("RGB")

    elif pixel_format.value == GPF.fmt_PFNC_YUV422_8.value:
        print("YUV422_8_YUYV mode")

        im = YUV422_YUYV_Decoder(im_size, im_addr.contents)

        # [Optional] Convert the image from YCbCr to RGB
        # im = im.convert("RGB")

    else:
        print("Mono8 mode")
        im = Image.frombuffer("L", im_size, im_addr.contents, "raw", "L", 0, 1)

    return im


def YUV422_Decoder(im_size, buffer, format):
    """Decode the buffer which is in YUV422 format.
    Parameters:
        im_size [in] imagesize (width,height)
        buffer [in]  buffer which contain yuv data
        format [in]  the position of each componants of UYVY, YUYV etc.
    Returns:
        Pillow image in YCbCr mode."""

    y = []
    cb = []
    cr = []

    for ptr in range(0, len(buffer), 4):
        # According to the format, assign values to each componant
        # It could be UYVY or YUYV
        U0 = buffer[ptr + format["U0"]]
        Y0 = buffer[ptr + format["Y0"]]
        V0 = buffer[ptr + format["V0"]]
        Y1 = buffer[ptr + format["Y1"]]

        # upsampling
        # pixel n+0
        y.append(Y0)
        cb.append(U0)
        cr.append(V0)

        # pixel n+1
        y.append(Y1)
        cb.append(U0)
        cr.append(V0)

    y = Image.frombuffer("L", im_size, bytes(y), "raw", "L", 0, 1)
    cb = Image.frombuffer("L", im_size, bytes(cb), "raw", "L", 0, 1)
    cr = Image.frombuffer("L", im_size, bytes(cr), "raw", "L", 0, 1)

    im = Image.merge("YCbCr", (y, cb, cr))

    return im


def YUV422_UYVY_Decoder(im_size, buffer):
    """The wrapper of YUV422_Decoder()
    Parameters:
        im_size [in] imagesize (width,height)
        buffer [in]  buffer which contain yuv data
    Returns:
        Pillow image in YCbCr mode.
    Notes:
        UYVY Images are stored top-down, so the upper left pixel starts at byte 0.
        Each 4 bytes represent the color for 2 neighboring pixels:
        [ U0 | Y0 | V0 | Y1 ]
        Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
        U0 and V0 is the color of both pixels."""

    # First byte is U0, Second byte is Y0, etc.
    format = {"U0": 0, "Y0": 1, "V0": 2, "Y1": 3}
    im = YUV422_Decoder(im_size, buffer, format)

    return im


def YUV422_YUYV_Decoder(im_size, buffer):
    """The wrapper of YUV422_Decoder()
    Parameters:
        im_size [in] imagesize (width,height)
        buffer [in]  buffer which contain yuv data
    Returns:
        Pillow image in YCbCr mode.
    Notes:
        YUYV Images are stored top-down, so the upper left pixel starts at byte 0.
        Each 4 bytes represent the color for 2 neighboring pixels:
        [ Y0 | U0 | Y1 | V0 ]
        Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
        U0 and V0 is the color of both pixels."""

    # First byte is Y0, Second byte is U0, etc.
    format = {"Y0": 0, "U0": 1, "Y1": 2, "V0": 3}
    im = YUV422_Decoder(im_size, buffer, format)

    return im


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
        return (self.width * self.height) * self.pixel_bitsize

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
        print(f"init transfer status: {status}")

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
            if status != 0:
                print(f"Next frame is not available. Status: {status}")
                continue

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

        width_name, height_name = CameraInfo.dimen_names()

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
    
    for i in range(10):
        frames = list(camera.read())

        if len(frames) > 0:
            frame = frames[0]
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", frame)
            cv2.waitKey(0)
    

    camera.release()
    pygigev.GevApiUninitialize()
