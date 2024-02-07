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
__iDOT = bytes("iDOT", "ascii")
headerlist = [__bKGD, __cHRM, __dSIG, __eXIf, __gAMA, __hIST, __iCCP,
              __iTXt, __pHYs, __sBIT, __sPLT, __sRGB, __sTER, __tEXt,
              __tIME, __tRNS, __zTXt, __iDOT]

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

def __defilter(decompressed, height, width, true_width, byte_per_pixel):
    decompressed = [k for k in decompressed]  # Make it an integer list
    im_matrix = []
    for i in range(height):
        f_type = decompressed[i * true_width]
        row = []
        if f_type == 0:  # No filtering
            for j in range(width):
                pixel = []
                for k in range(byte_per_pixel):
                    pixel.append(decompressed[k + (i * true_width + 1) + j * byte_per_pixel])
                    #decompressed[k + (i * true_width + 1) + j * byte_per_pixel] = pixel[-1]
                row.append(pixel)
        elif f_type == 1:  # Sub filtering
            a = 0
            for j in range(width):
                pixel = []
                for k in range(byte_per_pixel):
                    val = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                    pixel.append((val - a) % 256)
                    decompressed[k + (i * true_width + 1) + j * byte_per_pixel] = pixel[-1]
                    a = val % 256
                row.append(pixel)
        elif f_type == 2:  # Up filtering
            if i == 0:  # Will be the same as no filter since Prior(x) = 0 for all x
                for j in range(width):
                    pixel = []
                    for k in range(byte_per_pixel):
                        pixel.append(decompressed[k + (i * true_width + 1) + j * byte_per_pixel])
                        decompressed[k + (i * true_width + 1) + j * byte_per_pixel] = pixel[-1]
                    row.append(pixel)
            else:
                for j in range(width):
                    pixel = []
                    for k in range(byte_per_pixel):
                        val1 = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                        val2 = decompressed[k + ((i - 1) * true_width + 1) + j * byte_per_pixel]
                        pixel.append((val1 + val2) % 256)
                        decompressed[k + (i * true_width + 1) + j * byte_per_pixel] = pixel[-1]
                    row.append(pixel)
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
                        decompressed[k + (i * true_width + 1) + j * byte_per_pixel] = pixel[-1]
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
                        decompressed[k + (i * true_width + 1) + j * byte_per_pixel] = pixel[-1]
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
                        pixel[-1] = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
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
                        pixel[-1] = decompressed[k + (i * true_width + 1) + j * byte_per_pixel]
                    row.append(pixel)
        im_matrix.append(row)
    logger.debug("Raw matrix loaded.")

    mlist = [0 for k in range(byte_per_pixel)]
    rowlist = [[] for k in range(byte_per_pixel)]
    for p in im_matrix[0]:
        for k in range(byte_per_pixel):
            rowlist[k].append(p[k])

    for k in range(byte_per_pixel):
        mlist[k] = Matrix(Vector(*(rowlist[k])))

    for row in im_matrix[1:]:
        rowlist = [[] for l in range(byte_per_pixel)]
        for p in row:
            for k in range(byte_per_pixel):
                rowlist[k].append(p[k])
        for k in range(byte_per_pixel):
            mlist[k].append(Vector(*rowlist[k]))
    logger.debug("Matrix reformatted to a Tensor.")
    return Tensor(*mlist)


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

            # Defiltering will be the same no matter the load method
            im_matrix = __defilter(decompressed, height, width, true_width, byte_per_pixel)
            # Above is already a Tensor object.

            if method == LOAD_ASIS:
                pass  # We will literally do nothing
            elif method == LOAD_RGB:
                if colotypename == "GRAYSCALE" or colotypename == "GRAYSCALE WITH ALPHA" or colotypename == "TRUE RGB":
                    pass
                elif colotypename == "RGB WITH ALPHA":
                    opacity = pow(2, int.from_bytes(bit_depth, "big")) - 1
                    for matrix in im_matrix[:-1]:
                        for k in range(height):
                            for l in range(width):
                                matrix.values[k][l] *= (1 - im_matrix.values[-1][k][l]) / opacity
                                matrix.values[k][l] %= 256

                    return Image(Tensor(*(im_matrix[:-1])), "PNG", "TRUE RGB", method)
            elif method == LOAD_BGR:
                if colotypename == "GRAYSCALE":
                    Image(Tensor(im_matrix.values[0], im_matrix.values[0], im_matrix.values[0]), "PNG", "TRUE BGR", method)
                elif colotypename == "GRAYSCALE WITH ALPHA":
                    opacity = pow(2, int.from_bytes(bit_depth, "big")) - 1
                    for row in im_matrix.values[0]:
                        for k in range(im_matrix.dimension[1]):
                            row[k] *= (1 - row[k]) / opacity
                            row[k] %= 256

                elif colotypename == "TRUE RGB":
                    return Image(Tensor(*(im_matrix.values[::-1])), "PNG", "TRUE BGR", method)

                elif colotypename == "RGB WITH ALPHA":
                    opacity = pow(2, int.from_bytes(bit_depth, "big")) - 1
                    for matrix in im_matrix[:-1]:
                        for k in range(height):
                            for l in range(width):
                                matrix.values[k][l] *= (1 - im_matrix.values[-1][k][l]) / opacity
                                matrix.values[k][l] %= 256
                    return Image(Tensor(*((im_matrix.values[:-1])[::-1])), "PNG", "TRUE BGR", method)
            elif method == LOAD_GRAYSCALE:
                if colotypename == "GRAYSCALE":
                    pass

                elif colotypename == "TRUE RGB":
                    new_im_matrix = (  0.299 * im_matrix.values[0]
                                     + 0.587 * im_matrix.values[1]
                                     + 0.114 * im_matrix.values[2]).map(lambda x: x % 256)

                    return Image(Tensor(new_im_matrix), "PNG", "GRAYSCALE", method)

                elif colotypename == "RGB WITH ALPHA":
                    opacity = pow(2, int.from_bytes(bit_depth, "big")) - 1
                    new_im_matrix = (  0.299 * im_matrix.values[0]
                                     + 0.587 * im_matrix.values[1]
                                     + 0.114 * im_matrix.values[2])
                    for k in range(height):
                        for l in range(width):
                            new_im_matrix.values[k][l] *= (1 - im_matrix.values[-1][k][l]) / opacity
                            new_im_matrix.values[k][l] %= 256
                    return Image(Tensor(new_im_matrix), "PNG", "GRAYSCALE", method)

                elif colotypename == "GRAYSCALE WITH ALPHA":
                    opacity = pow(2, int.from_bytes(bit_depth, "big")) - 1
                    for k in range(height):
                        for l in range(width):
                            im_matrix.values[0][k][l] *= (1 - im_matrix.values[1][k][l]) / opacity
                            im_matrix.values[0][k][l] %= 256
                    return Image(Tensor(im_matrix.values[0]), "PNG", "GRAYSCALE", method)

            return Image(im_matrix, "PNG", colotypename, method)
        else:
            raise FormatError()

def imsave(path, img):
    if not isinstance(img, Image): raise ArgTypeError("Incorrect variable type for image, must be of type Image.")

    ihdr_header = bytes()
    ihdr_header += int.to_bytes(img.matrix.dimension[2], 4, "big")  #width
    ihdr_header += int.to_bytes(img.matrix.dimension[1], 4, "big")  #height
    ihdr_header += int.to_bytes(8, 1, "big")  #bit-depth (yes, always 8 when writing for now)
    # 8 is a supported bit depth for all image types

    # Color type
    if img.colortype == "GRAYSCALE":
        ihdr_header += __grayscale
    elif img.colortype == "GRAY WITH ALPHA":
        ihdr_header += __grayalpha
    elif img.colortype == "TRUE RGB":
        ihdr_header += __rgbtrue
    elif img.colortype == "RGB WITH ALPHA":
        ihdr_header += __rgba
    else:
        logger.warning(f"Unrecognized color type: {img.colortype}")
        return

    ihdr_header += int.to_bytes(0, 1, "big")  # Compression type, always 0
    ihdr_header += int.to_bytes(0, 1, "big")  # Filter method is again always 0
    ihdr_header += int.to_bytes(0, 1, "big")  # No interlace is supported yet
    ihdr_checksum = int.to_bytes(zlib.crc32(__IHDR + ihdr_header), 4, "big")
    # 17 bytes here
    ihdr_length = int.to_bytes(13, 4, "big")
    ihdr_data = ihdr_length + __IHDR + ihdr_header + ihdr_checksum  # IHDR chunk done.

    # Since we do not support __indexed format yet, our images do not
    # need to have a PLTE chunk. It is only a must for __indexed; color type 3.

    image_data = bytes()
    logger.debug("IHDR ready, begin IDAT preparation.")
    for i in range(img.matrix.dimension[1]):
        image_data += int.to_bytes(0, 1, "big")  # We will not apply filtering
        for j in range(img.matrix.dimension[2]):
            for layer in range(img.matrix.dimension[0]):
                image_data += int.to_bytes(img.matrix.values[layer].values[i][j], 1, "big")
    logger.debug("Begin compression.")
    image = __IDAT + zlib.compress(image_data)
    logger.debug("End compression.")
    image_checksum = int.to_bytes(zlib.crc32(image), 4, "big")
    idat_length = int.to_bytes(len(image) - 4, 4, "big")
    idat_data = idat_length + image + image_checksum
    logger.debug("IDAT ready.")
    # IDAT chunk done.

    end_length = int.to_bytes(0, 4, "big")  # Name + crc checksum = 4 + 4 = 8
    end_data = end_length + __IEND
    end_checksum = int.to_bytes(zlib.crc32(end_data), 4, "big")
    end = end_data + end_checksum

    with open(path, "wb") as file:
        file.write(__PNG + ihdr_data + idat_data + end)

    logger.info(f"Image saved at {path}")

class Image:

    def __init__(self, matrix, imtype, colortype, loadtype=LOAD_RGB):
        self.matrix = matrix
        self.imtype = imtype
        self.loadtype = loadtype
        self.colortype = colortype

    def __repr__(self):
        return f"<{self.matrix.dimension} {self.imtype} image loaded as load type {self.colortype}>"

    def __str__(self):
        return str(self.matrix)

    def describe(self):
        print(self.__repr__())



