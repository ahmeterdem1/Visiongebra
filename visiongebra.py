from vectorgebra import *
import os
import zlib

LOAD_RGB = 0
LOAD_GRAYSCALE = 1
LOAD_BGR = 2
LOAD_ASIS = 3

# Constants for the png file format

__PNG = int.to_bytes(0x89504E470D0A1A0A, 8, "big")

__IHDR = bytes("IHDR", "ascii")
__PLTE = bytes("PLTE", "ascii")
__IDAT = bytes("IDAT", "ascii")
__IEND = bytes("IEND", "ascii")

__bKGD = bytes("bKGD", "ascii")
__cHRM = bytes("cHRM", "ascii")
__dSIG = bytes("dSIG", "ascii")
__eXIf = bytes("eXIf", "ascii")
__gAMA = bytes("gAMA", "ascii")
__hIST = bytes("hIST", "ascii")
__iCCP = bytes("iCCP", "ascii")
__iTXt = bytes("iTXt", "ascii")
__pHYs = bytes("pHYs", "ascii")
__sBIT = bytes("sBIT", "ascii")
__sPLT = bytes("sPLT", "ascii")
__sRGB = bytes("sRGB", "ascii")
__sTER = bytes("sTER", "ascii")
__tEXt = bytes("tEXt", "ascii")
__tIME = bytes("tIME", "ascii")
__tRNS = bytes("tRNS", "ascii")
__zTXt = bytes("zTXt", "ascii")
headerlist = [__bKGD, __cHRM, __dSIG, __eXIf, __gAMA, __hIST, __iCCP,
              __iTXt, __pHYs, __sBIT, __sPLT, __sRGB, __sTER, __tEXt,
              __tIME, __tRNS, __zTXt]

__grayscale = int.to_bytes(0, 1, "big")  # 0
__rgbtrue = int.to_bytes(2, 1, "big")  # 2
__indexed = int.to_bytes(3, 1, "big")  # 3
__grayalpha = int.to_bytes(4, 1, "big")  # 4
__rgba = int.to_bytes(6, 1, "big")  # 6

__crcp = int.to_bytes(0xffffffff, 4, "big")  # CRC Polynomial for the PNG format


class FormatError(Exception):

    def __init__(self):
        super().__init__("Image format not supported.")

class ChunkError(Exception):

    def __init__(self, hint: str = ''):
        super().__init__(f"Chunk error when reading image{': ' + hint if hint else '.'}")


def __reverse(n):
    rev = 0
    for k in range(8):
        if n > 0:
            rev <<= 1
            if n & 1 == 1:
                rev ^= 1
            n >>= 1
        else:
            rev <<= 1
    return rev

def __crc32(data, checksum):
    data += bytes(4)  # empty 32 bits appended to the end of the data

    temp = data[:5]
    counter = 5

    n = len(data)
    null = bytes()

    # When exiting the below loop, temp will always have a length of 5 bytes
    while True:
        if counter == n:
            break
        elif temp[0] == null:
            temp = temp[1:]  # Reduce its length to 4 bytes
            temp += data[counter]  # Increase the length of it to 5 bytes again
            counter += 1
        else:
            crcp = 0xffffffff
            first = int.from_bytes(temp[0], "big")
            shift = __reverse(first)
            while shift > 0:
                shift >>= 1
                crcp <<= 1

            # crcp is ready to be used
            first = int.from_bytes(temp, "big")  # Reuse the variables
            temp = int.to_bytes(first - crcp, 5, "big")


    if temp[1:] != checksum:
        raise ChunkError("Chunk corrupted, did not pass CRC check")

def __paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)

    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def imread(path, method=LOAD_RGB):
    if not os.path.exists(path):
        raise FileNotFoundError()

    total_data = bytes()
    with open(path, "rb") as image:
        signature = image.read(8)
        total_data += signature
        if signature == __PNG:
            logger.debug("Image identified as PNG.")

            # First chunk must be IHDR
            chunk_length = image.read(4)
            chunk_signature = image.read(4)
            total_data += chunk_length
            total_data += chunk_signature

            if chunk_signature != __IHDR: raise ChunkError("First chunk must be IHDR.")

            ihdr_data = image.read(int.from_bytes(chunk_length, "big"))
            checksum = image.read(4)
            if zlib.crc32(chunk_signature + ihdr_data) != int.from_bytes(checksum, "big"):
                raise ChunkError("Chunk corrupted, did not pass CRC check")
            total_data += ihdr_data
            total_data += checksum

            width = int.from_bytes(ihdr_data[:4], "big")
            height = int.from_bytes(ihdr_data[4:8], "big")
            bit_depth = ihdr_data[8:9]
            color_type = ihdr_data[9:10]

            byte_per_pixel = 0
            colotypename = ""

            if color_type == __grayscale:
                byte_per_pixel = 1
                colotypename = "GRAYSCALE"
            elif color_type == __grayalpha:
                byte_per_pixel = 2
                colotypename = "GRAYSCALE WITH ALPHA"
            elif color_type == __rgbtrue:
                byte_per_pixel = 3
                colotypename = "TRUE RGB"
            elif color_type == __rgba:
                byte_per_pixel = 4
                colotypename = "RGB WITH ALPHA"
            elif color_type == __indexed:
                logger.warning("Indexed color type not supported yet, terminating image reading.")
                return
            else:
                raise ChunkError(f"Unidentified color type: {str(color_type, 'ascii')}")

            compression_method = ihdr_data[10:11]

            if int.from_bytes(compression_method, "big"):
                raise ChunkError("Invalid compression method at chunk IHDR.")

            filter_method = ihdr_data[11:12]
            if int.from_bytes(filter_method, "big"):
                raise ChunkError("Invalid filter method at chunk IHDR.")

            interlace = ihdr_data[12:13]
            if interlace != int.to_bytes(0, 1, "big")\
                and interlace != int.to_bytes(1, 1, "big"):
                raise ChunkError("Invalid interlace at chunk IHDR.")

            #total_data += width + height + bit_depth + color_type + compression_method + filter_method + interlace

            image_data = bytes()
            PLTEfound = False
            while True:
                current_chunk_length = image.read(4)
                current_chunk = image.read(4)

                if current_chunk == __IDAT:
                    current_chunk_data = image.read(int.from_bytes(current_chunk_length, "big"))
                    checksum = image.read(4)
                    #__crc32(current_chunk + current_chunk_data, checksum)
                    if zlib.crc32(current_chunk + current_chunk_data) != int.from_bytes(checksum, "big"):
                        raise ChunkError("Chunk corrupted, did not pass CRC check")

                    image_data += current_chunk_data
                    total_data += current_chunk_length + current_chunk + current_chunk_data + checksum

                elif current_chunk == __PLTE:
                    PLTEfound = True
                    if color_type == int.to_bytes(0, 1, "big")\
                        or color_type == int.to_bytes(4, 1, "big"):
                        raise ChunkError("Chunk PLTE must not appear.")

                    current_chunk_data = image.read(int.from_bytes(current_chunk_length, "big"))
                    checksum = image.read(4)
                    #__crc32(current_chunk + current_chunk_data, checksum)
                    if zlib.crc32(current_chunk + current_chunk_data) != int.from_bytes(checksum, "big"):
                        raise ChunkError("Chunk corrupted, did not pass CRC check")
                    total_data += current_chunk_length + current_chunk + current_chunk_data + checksum

                elif current_chunk == __IEND:
                    current_chunk_data = image.read()
                    total_data += current_chunk_length + current_chunk + current_chunk_data
                    break

                elif current_chunk in headerlist:
                    # We basically ignore those chunks, at least for now.
                    current_chunk_data = image.read(int.from_bytes(current_chunk_length, "big"))
                    checksum = image.read(4)
                    #__crc32(current_chunk + current_chunk_data, checksum)
                    if zlib.crc32(current_chunk + current_chunk_data) != int.from_bytes(checksum, "big"):
                        raise ChunkError("Chunk corrupted, did not pass CRC check")
                    total_data += current_chunk_length + current_chunk + current_chunk_data + checksum
                else:
                    raise ChunkError(f"Unidentified chunk: {str(current_chunk, encoding='ascii')}")

            if (not PLTEfound) and color_type == int.to_bytes(3, 1, "big"):
                raise ChunkError("No PLTE chunk found.")

            decompressed = zlib.decompress(image_data)
            true_width = width * byte_per_pixel + 1

            if method == LOAD_ASIS:
                im_matrix = []
                for i in range(height):
                    f_type = decompressed[i * true_width]
                    row = []
                    if f_type == 0:  # No filtering
                        for j in range(width):
                            pixel = []
                            for k in range(byte_per_pixel):
                                pixel.append(decompressed[k + (i * true_width + 1) + j * byte_per_pixel])
                            row.append(pixel)
                    elif f_type == 1:  # Sub filtering
                        a = 0
                        for j in range(width):
                            pixel = []
                            for k in range(byte_per_pixel):
                                val = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                                pixel.append((val - a) % 256)
                                a = val % 256
                            row.append(pixel)
                    elif f_type == 2:  # Up filtering
                        if i == 0:  # Will be the same as no filter since Prior(x) = 0 for all x
                            for j in range(width):
                                pixel = []
                                for k in range(byte_per_pixel):
                                    pixel.append(decompressed[k + (i * true_width + 1) + j * byte_per_pixel])
                                row.append(pixel)
                        else:
                            for j in range(width):
                                pixel = []
                                for k in range(byte_per_pixel):
                                    val1 = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                                    val2 = decompressed[k + ((i - 1) * true_width + 1) + j * byte_per_pixel]
                                    pixel.append((val1 + val2) % 256)
                                row.append(pixel)
                        pass
                    elif f_type == 3:
                        if i == 0:
                            a, b = 0, 0
                            for j in range(width):
                                pixel = []
                                for k in range(byte_per_pixel):
                                    if k + j:
                                        a = decompressed[k + (i * true_width) + j * byte_per_pixel]
                                    this = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                                    pixel.append((this + (a + b) % 256) % 256)
                                row.append(pixel)
                        else:
                            a, b = 0, 0
                            for j in range(width):
                                pixel = []
                                for k in range(byte_per_pixel):
                                    if k + j:
                                        a = decompressed[k + (i * true_width) + j * byte_per_pixel]
                                    b = decompressed[k + ((i - 1) * true_width + 1) + j * byte_per_pixel]
                                    this = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                                    pixel.append((this + (a + b) % 256) % 256)
                                row.append(pixel)
                    elif f_type == 4:
                        if i == 0:
                            a, b, c = 0, 0, 0
                            for j in range(width):
                                pixel = []
                                for k in range(byte_per_pixel):
                                    if j + k:
                                        a = decompressed[k + (i * true_width) + j * byte_per_pixel]
                                    this = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                                    this = (this + __paeth(a, b, c)) % 256
                                    pixel.append(this)
                                row.append(pixel)
                        else:
                            a, b, c = 0, 0, 0

                            for j in range(width):
                                pixel = []
                                for k in range(byte_per_pixel):
                                    b = decompressed[k + ((i - 1) * true_width + 1) + j * byte_per_pixel]
                                    if j + k:
                                        a = decompressed[k + (i * true_width) + j * byte_per_pixel]
                                        c = decompressed[k + ((i - 1) * true_width) + j * byte_per_pixel]
                                    this = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                                    this = (this + __paeth(a, b, c)) % 256
                                    pixel.append(this)
                                row.append(pixel)

                    im_matrix.append(row)
            elif method == LOAD_RGB:
                pass
            elif method == LOAD_BGR:
                pass
            elif method == LOAD_GRAYSCALE:
                pass

            return Image(im_matrix, "PNG", colotypename, method)


        else:
            raise FormatError()

def imsave(path, img):
    if not isinstance(img, Image): raise ArgTypeError("Incorrect variable type for image, must be visiongebra.Image")

    logger.info(f"Image saved at {path}.{img.imtype}")

class Image:

    def __init__(self, matrix, imtype, colortype, loadtype=LOAD_RGB):
        self.matrix = matrix
        self.imtype = imtype
        self.loadtype = loadtype
        self.colortype = colortype

    def __repr__(self):
        return f"<{0} {self.imtype} image loaded as load type {self.loadtype}>"

    def __str__(self):
        return str(self.matrix)

