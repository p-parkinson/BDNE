# matlab_serialise.py
# Functions to mimic hlp_serialise and hlp_deserialise packages from MATLAB
# Author: Patrick Parkinson <patrick.parkinson@manchester.ac.uk>
# Created: 30/06/2020
#
# Only a limited subset of commands introduced.
import numpy as np

classes = [np.double, np.single, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
           np.uint64]
sizes = [8, 4, 1, 1, 2, 2, 4, 4, 8, 8]


def deserialise(value: bytes, return_pos: bool = False):
    """Deserialise bytes to numpy array

    This function is complementary to the hlp_deserialise.m function available in MATLAB, which is described at
    https://uk.mathworks.com/matlabcentral/fileexchange/34564-fast-serialize-deserialize.

    It takes a byte string in a specific format and returns a numpy array with appropriate data type.
    Only a subset of types are implemented. """
    pos = 0
    if value[0] in range(1, 11):
        # Interpret as a scalar
        sz = sizes[value[0] - 1]
        dt = np.frombuffer(value[1:(1 + sz)], classes[value[0] - 1])
        pos = 1 + sz
    elif value[0] in range(17, 26):
        # Interpret as a simple numeric array
        sz = sizes[value[0] - 17]
        ndims = np.uint8(value[1])
        dims = np.frombuffer(value[2:(2 + ndims * 4)], 'uint32')
        pos = 2 + ndims * 4
        nbytes = sz * np.prod(dims)
        dt = np.frombuffer(value[pos:(pos + nbytes)], classes[value[0] - 17])
        dt = np.squeeze(np.reshape(dt, dims[::-1]))
        pos += nbytes
    elif value[0] == 128:
        # Interpret as a struct array
        nfields = np.frombuffer(value[1:5], dtype='uint32', count=1)[0]
        fnLengths = np.frombuffer(value[5:(5 + nfields * 4)], dtype='uint32', count=nfields)
        pos = (5 + nfields * 4)
        fnchars = value[pos:pos + sum(fnLengths)]
        pos += sum(fnLengths)
        ndms = np.frombuffer(value[pos:pos + sum(fnLengths) + 4], dtype='uint32', count=1)[0]
        pos += 4
        dms = np.frombuffer(value[pos:pos + ndms * 4], 'uint32')
        pos += ndms * 4
        fieldnames = (fnchars[i[0]:i[1]].decode("utf-8") for i in np.transpose(
            [np.cumsum(np.insert(fnLengths, 0, 0))[:-1], np.cumsum(np.insert(fnLengths, 0, 0))[1:]]))
        if value[pos]:
            # Via cell
            dt, p = deserialise(value[pos + 1:], return_pos=True)
            pos += p
            dt = {i: j for (i, j) in zip(fieldnames, dt)}
        else:
            # Otherwise
            # TODO: Implement Cell
            raise NotImplementedError('Other struct type not implemented')
    elif value[0] in range(33, 39):
        # Is a MATLAB cell
        kind = value[0]
        if kind == 33:
            # Arbitrary cell array
            ndms = value[1]
            # Dimension
            dms = np.frombuffer(value[2:2 + ndms * 4], dtype='uint32', count=ndms)
            pos_initial = 2 + ndms * 4
            dt = tuple()
            for i in range(np.product(dms)):
                d, pos = deserialise(value[pos_initial:], return_pos=True)
                pos_initial += pos
                dt += (d,)
        else:
            raise NotImplementedError('Other cell types not implemented')
    else:
        # unknown
        # TODO: Implement other types. strings etc.
        dt = value
        raise NotImplementedError('Type {} not implemented'.format(np.uint8(value[0])))
    if return_pos:
        return dt, pos
    else:
        return dt


def serialise(value: np.ndarray) -> bytes:
    """Serialise numpy array to a byte string

    This function is complementary to the hlp_serialise.m function available in MATLAB, which is described at
    https://uk.mathworks.com/matlabcentral/fileexchange/34564-fast-serialize-deserialize.

    It takes a numpy array of any basic datatype, and returns a byte string in a specific format.
    Only a subset of types are implemented. """
    datatype = classes.index(value.dtype)
    element_size = sizes[datatype]
    num_elements = np.size(value)
    if num_elements == 1:
        # Scalar
        obuf = np.zeros(1 + element_size, dtype=np.uint8)
        obuf[0] = datatype + 1
        obuf[1:] = [np.uint8(i) for i in value.tobytes()]
    elif num_elements > 1:
        # Array
        dims = value.shape[::-1]
        ndims = len(dims)
        obuf = np.zeros(1 + 1 + (ndims * 4) + (num_elements * element_size), dtype=np.uint8)
        # Record data type
        obuf[0] = datatype + 17
        # Record number of dimensions (as 8 bit)
        obuf[1] = ndims
        # Record dimensions (as 32bit)
        obuf[2:(2 + ndims * 4)] = np.reshape(
            [[np.uint8(i) for i in np.uint32(b).tobytes()] for b in dims],
            (1, 4 * ndims))
        # Record data
        obuf[(2 + ndims * 4):] = [np.uint8(i) for i in value.tobytes()]
    else:
        # Unknown
        # TODO: Implement other data type conversions - complex numbers etc.
        raise (NotImplementedError("Unknown data type in Serialise"))
    return obuf.tobytes()
