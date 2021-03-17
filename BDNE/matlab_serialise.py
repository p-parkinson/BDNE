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


def deserialise(input_bytes: bytes, return_cursor: bool = False):
    """Deserialise bytes to numpy array

    This function is complementary to the hlp_deserialise.m function available in MATLAB, which is described at
    https://uk.mathworks.com/matlabcentral/fileexchange/34564-fast-serialize-deserialize.

    It takes a byte string in a specific format and returns a numpy array with appropriate data type.
    Only a subset of types are implemented. """

    # Used to hold current position (for iterative use)
    cursor = 0
    # Select type based on first byte
    if input_bytes[0] in range(1, 11):
        # Interpret as a scalar
        sz = sizes[input_bytes[0] - 1]
        # Pull from buffer, with type given by the first byte
        dt = np.frombuffer(input_bytes[1:(1 + sz)], classes[input_bytes[0] - 1])
        cursor = 1 + sz
    elif input_bytes[0] in range(17, 26):
        # Interpret as a simple numeric array
        sz = sizes[input_bytes[0] - 17]
        # Get number of dimension
        number_of_dimensions = np.uint8(input_bytes[1])
        # Get the length of each dimension
        dims = np.frombuffer(input_bytes[2:(2 + number_of_dimensions * 4)], 'uint32')
        # Move the cursor forwards
        cursor = np.int(2 + number_of_dimensions * 4)
        # Work out how many data bytes are required
        number_of_bytes = np.int(sz * np.prod(dims))
        # Get data from buffer
        dt = np.frombuffer(input_bytes[cursor:(cursor + number_of_bytes)], classes[input_bytes[0] - 17])
        # Change the shape of the array as appropriate
        dt = np.squeeze(np.reshape(dt, dims[::-1]))
        # Move cursor
        cursor += number_of_bytes
    elif input_bytes[0] == 128:
        # Interpret as a structure array
        number_of_fields = np.frombuffer(input_bytes[1:5], dtype='uint32', count=1)[0]
        # Get each field name length (in characters)
        field_name_lengths = np.frombuffer(
            input_bytes[5:(5 + number_of_fields * 4)], dtype='uint32', count=number_of_fields)
        # Move cursor
        cursor = np.int(5 + number_of_fields * 4)
        # Get the actual field name characters
        field_name_chars = input_bytes[cursor:cursor + sum(field_name_lengths)]
        # Move cursor
        cursor += sum(field_name_lengths)
        # Get the number of dimensions
        number_of_dimensions = np.frombuffer(
            input_bytes[cursor:cursor + sum(field_name_lengths) + 4], dtype='uint32', count=1)[0]
        # Move cursor
        cursor += 4
        # Get the actual dimensions length
        # UNUSED: dimensions = np.frombuffer(value[cursor:cursor + number_of_dimensions * 4], 'uint32')
        # Move cursor
        cursor += number_of_dimensions * 4
        # Separate the fieldnames, given a list of start positions
        fieldnames = (
            field_name_chars[i[0]:i[1]].decode("utf-8") for i in
            np.transpose(
                [np.cumsum(np.insert(field_name_lengths, 0, 0))[:-1],
                 np.cumsum(np.insert(field_name_lengths, 0, 0))[1:]
                 ]
            )
        )
        # Next, interpret the data
        if input_bytes[cursor]:
            # Data is stored as a cell - convert the cell back
            dt, p = deserialise(input_bytes[cursor + 1:], return_cursor=True)
            # Move the cursor
            cursor += p
            # Convert cell to a dict array to be more "struct"-like
            dt = {i: j for (i, j) in zip(fieldnames, dt)}
        else:
            # Otherwise data is not stored as a cell
            # TODO: Implement Cell
            raise NotImplementedError('Struct types other than cell2struct are not implemented')
    elif input_bytes[0] in range(33, 39):
        # Is a MATLAB cell
        kind = input_bytes[0]
        # There's lots of cell types (6). All need implementing.
        if kind == 33:
            # Arbitrary cell array
            number_of_dimensions = input_bytes[1]
            # Dimension
            dimensions = np.frombuffer(
                input_bytes[2:2 + number_of_dimensions * 4], dtype='uint32', count=number_of_dimensions)
            # Move cursor
            pos_initial = 2 + number_of_dimensions * 4
            # Output as a tuple
            dt = tuple()
            # Iterate over the possible dimensions
            for i in range(np.product(dimensions)):
                # Recursively call deserialise on each part of the cell array
                d, cursor = deserialise(input_bytes[pos_initial:], return_cursor=True)
                # Move cursor
                pos_initial += cursor
                # Add to tuple
                dt += (d,)
        else:
            # TODO: Implement other cell types
            raise NotImplementedError('Cell types other than "arbitrary cell array" not implemented')
    else:
        # unknown type
        # TODO: Implement other types. strings etc.
        raise NotImplementedError('Type {} not yet implemented'.format(np.uint8(input_bytes[0])))
    # Check if a the "return position" is required.
    # This is a bit of a workaround to run in both normal and iterative modes
    if return_cursor:
        return dt, cursor
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
        output_buffer = np.zeros(1 + element_size, dtype=np.uint8)
        output_buffer[0] = datatype + 1
        output_buffer[1:] = [np.uint8(i) for i in value.tobytes()]
    elif num_elements > 1:
        # Array
        dims = value.shape[::-1]
        number_of_dimensions = len(dims)
        output_buffer = np.zeros(1 + 1 + (number_of_dimensions * 4) + (num_elements * element_size), dtype=np.uint8)
        # Record data type
        output_buffer[0] = datatype + 17
        # Record number of dimensions (as 8 bit)
        output_buffer[1] = number_of_dimensions
        # Record dimensions (as 32bit)
        output_buffer[2:(2 + number_of_dimensions * 4)] = np.reshape(
            [[np.uint8(i) for i in np.uint32(b).tobytes()] for b in dims],
            (1, 4 * number_of_dimensions))
        # Record data
        output_buffer[(2 + number_of_dimensions * 4):] = [np.uint8(i) for i in value.tobytes()]
    else:
        # Unknown
        # TODO: Implement other data type conversions - complex numbers etc.
        raise (NotImplementedError("Unknown data type in Serialise"))
    return output_buffer.tobytes()
