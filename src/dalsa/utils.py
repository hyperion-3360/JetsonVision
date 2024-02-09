# =======================================================================

from functools import reduce

import sys

from PIL import Image

# Get the common support code for the GigE-V Framework for Linux
# (Change this if directory structure is changed).
import os
sys.path.append(os.path.dirname(__file__) + "/gigev_common")
from pygigev import GevPixelFormats as GPF


# Utilties
def ipAddr_from_string(s):
    "Convert dotted IPv4 address to integer."
    return reduce(lambda a, b: a << 8 | b, map(int, s.split(".")))


def ipAddr_to_string(ip):
    "Convert 32-bit integer to dotted IPv4 address."
    return ".".join(map(lambda n: str(ip >> n & 0xFF), [24, 16, 8, 0]))


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

