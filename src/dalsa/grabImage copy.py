#!/usr/bin/env python3

# =======================================================================
#
# grabImage.py
#
# Example showing how to grab 2 images from the first camera
# detected and display one of them using a PIL Image.

import sys
import ctypes
import time

from PIL import Image

#
#
# Get the common support code for the GigE-V Framework for Linux
# (Change this if directory structure is changed).
import os

sys.path.append(os.path.dirname(__file__) + "/../gigev_common")

import pygigev  # includeded in ../gigev_common, DO NOT install from pip
from pygigev import GevPixelFormats as GPF


# =======================================================================
# Utilties
def ipAddr_from_string(s):
    "Convert dotted IPv4 address to integer."
    return reduce(lambda a, b: a << 8 | b, map(int, s.split(".")))


def ipAddr_to_string(ip):
    "Convert 32-bit integer to dotted IPv4 address."
    return ".".join(map(lambda n: str(ip >> n & 0xFF), [24, 16, 8, 0]))


#
# The basic program
def main():
    # Initialize the API
    pygigev.GevApiInitialize()

    # Allocate a maximum number of camera info structures.
    maxCameras = 16
    numFound = (ctypes.c_uint32)(0)
    camera_info = (pygigev.GEV_CAMERA_INFO * maxCameras)()

    # Get the camera list
    status = pygigev.GevGetCameraList(camera_info, maxCameras, ctypes.byref(numFound))
    if status != 0:
        print("Error ", status, "getting camera list - exitting")
        quit()

    if numFound.value == 0:
        print("No cameras found - exitting")
        quit()

    # Proceed
    print(numFound.value, " Cameras found")
    for camIndex in range(numFound.value):
        print("ip = ", ipAddr_to_string(camera_info[camIndex].ipAddr))

    # Select the first camera and open it.
    camIndex = 0
    print("Opening camera #", camIndex)
    handle = (ctypes.c_void_p)()
    status = pygigev.GevOpenCamera(
        camera_info[camIndex], pygigev.GevExclusiveMode, ctypes.byref(handle)
    )

    # Get the payload parameters
    print("Getting payload information :")
    payload_size = (ctypes.c_uint64)()
    pixel_format = (ctypes.c_uint32)()
    status = pygigev.GevGetPayloadParameters(
        handle, ctypes.byref(payload_size), ctypes.byref(pixel_format)
    )
    pixel_format_unpacked = pygigev.GevGetUnpackedPixelType(pixel_format)
    print(
        "status :",
        status,
        " payload_size : ",
        payload_size.value,
        " pixel_format = ",
        hex(pixel_format.value),
        " pixel_format_unpacked = ",
        hex(pixel_format_unpacked),
    )

    # Get the Width and Height (extra information)
    feature_strlen = (ctypes.c_int)(pygigev.MAX_GEVSTRING_LENGTH)
    unused = (ctypes.c_int)(0)
    if sys.version_info > (3, 0):
        width_name = b"Width"
        height_name = b"Height"
    else:
        width_name = "Width"
        height_name = "Height"

    width_str = ((ctypes.c_char) * feature_strlen.value)()
    height_str = ((ctypes.c_char) * feature_strlen.value)()
    status = pygigev.GevGetFeatureValueAsString(
        handle, width_name, unused, feature_strlen, width_str
    )
    status = pygigev.GevGetFeatureValueAsString(
        handle, height_name, ctypes.byref(unused), feature_strlen, height_str
    )

    print(
        "status :", status, " Width : ", width_str.value, " Height = ", height_str.value
    )

    # Allocate buffers to store images in (2 here).
    # (Handle cases where image is larger than payload due to pixel unpacking)
    numBuffers = 2
    print(" Allocate ", numBuffers, " buffers :")
    buffer_addresses = ((ctypes.c_void_p) * numBuffers)()

    bufsize = payload_size.value
    bufsize_unpacked = (
        int(width_str.value)
        * int(height_str.value)
        * pygigev.GevGetPixelSizeInBytes(pixel_format_unpacked)
    )
    if bufsize_unpacked > bufsize:
        bufsize = bufsize_unpacked
    print(" Using bufsize = ", bufsize)

    for bufIndex in range(numBuffers):
        temp = ((ctypes.c_char) * bufsize)()
        buffer_addresses[bufIndex] = ctypes.cast(temp, ctypes.c_void_p)
        print(" buffer_addresses[", bufIndex, "] = ", hex(buffer_addresses[bufIndex]))

    # Initialize a transfer (Asynchronous cycling)
    print("Init transfer :")
    status = pygigev.GevInitializeTransfer(
        handle, pygigev.Asynchronous, payload_size, numBuffers, buffer_addresses
    )

    # Grab images to fill the buffers
    numImages = numBuffers
    print("Snap ", numImages, " images :")
    status = pygigev.GevStartTransfer(handle, numImages)

    # Read the images out
    gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()
    displayed = 0

    for imgIndex in range(numImages):
        tmout = (ctypes.c_uint32)(1000)
        status = pygigev.GevWaitForNextFrame(
            handle, ctypes.byref(gevbufPtr), tmout.value
        )

        if status != 0:
            continue

        # Check img data status
        gevbuf = gevbufPtr.contents
        if gevbuf.status != 0:
            print("Img # ", imgIndex, " has status = ", gevbuf.status)
            continue

        print(
            "Img # ",
            imgIndex,
            " : id = ",
            gevbuf.id,
            " w = ",
            gevbuf.w,
            " h = ",
            gevbuf.h,
            " address = ",
            hex(gevbuf.address),
        )

        if displayed != 0:
            continue

        # Make a PIL image out of this frame (assume 8 bit Mono (also works for 8bit Bayer before decoding))
        displayed = 1
        im_size = (gevbuf.w, gevbuf.h)
        im_addr = ctypes.cast(
            gevbuf.address, ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size)
        )

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

        # Display the image
        # This creates a new window for each image that will persist even after the program exits!
        im.show()

    # Free the transfer
    print("Free transfer :")
    status = pygigev.GevFreeTransfer(handle)

    # Close the camera
    print("Close camera :")
    status = pygigev.GevCloseCamera(ctypes.byref(handle))

    # Uninitialize
    pygigev.GevApiUninitialize()


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


#
# Call the actual main function
#
main()
