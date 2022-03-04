"""
Creates a LOS component for the wireless MIMO channel. Either ULA (Uniform Linear Array) or UPA
    (Uniform Planar Array) structures can apply to the receiver or the transmitter. If ULA is considered at the receiver,
    the variables regarding Rx (rx_size, ant_spc_rx, AoA) are scalar. If UPA is considered, those variables are
    2D-tensors of size 2. The same scenario applies to the transmitter.
"""

import torch as tr

def LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len):
    """
    Inputs:
    (1) rx_size : The antenna structure at the receiver, either ULA (scalar) or UPA (1D-tensor with numel = 2).
                  If UPA structure is considered, the 0th and 1st elements are the number of rows (vertical direction)
                  and columns (horizontal direction) of the antenna structure, respectively. All elements must be real
                  and positive integers.
    (2) tx_size : The antenna structure at the transmitter, either ULA (scalar) or UPA (1D-tensor with numel = 2).
                  If UPA structure is considered, the 0th and 1st elements are the number of rows (vertical direction)
                  and columns (horizontal direction) of the antenna structure, respectively. All elements must be real
                  and positive integers.
    (3) ant_spc_rx : The spacing between antennas at the receiver, either ULA (scalar) or UPA (1D-tensor with numel = 2).
                     If UPA structure is considered, the 0th and 1st elements are distances between two adjacent
                     antennas in the vertical and horizontal direction, respectively. All elements must be real and
                     positive.
    (4) ant_spc_tx : The spacing between antennas at the transmitter, either ULA (scalar) or UPA (1D-tensor with numel
                     = 2). If UPA structure is considered, the 0th and 1st elements are distances between two adjacent
                     antennas in the vertical and horizontal direction, respectively. All elements must be real and
                     positive.
    (5) AoA : The angle of arrival at the receiver, either ULA (scalar) or UPA (1D-tensor with numel = 2). If UPA
              structure is considered, the 0th and 1st elements are the elevation (vertical) and azimuth angles
              (horizontal), respectively. All elements must be real and range from -pi to pi.
    (6) AoD : The angle of departure at the transmitter, either ULA (scalar) or UPA (1D-tensor with numel = 2). If UPA
              structure is considered, the 0th and 1st elements are the elevation (vertical) and azimuth angles
              (horizontal), respectively. All  elements must be real and range from -pi to pi.
    (7) wave_len : wave length of the RF signal, in meter. It must be real, scalar, and positive.
    """
    if tr.is_complex(rx_size) or tr.numel(rx_size) > 2 or tr.any(rx_size < 1) or tr.is_floating_point(rx_size):
        print('WARNING!!! input rx_size must be real, positive integer, and having one or two elements')
        print('your input rx_size = ', rx_size)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(tx_size) or tr.numel(tx_size) > 2 or tr.any(tx_size < 1) or tr.is_floating_point(tx_size):
        print('WARNING!!! input tx_size must be real, positive integer, and having one or two elements')
        print('your input tx_size = ', tx_size)
        raise ValueError('INPUT ERROR')
    ula = 1
    upa = 2
    if tr.numel(rx_size) == 1:
        rx_struct = ula
    else:
        rx_struct = upa

    if tr.numel(tx_size) == 1:
        tx_struct = ula
    else:
        tx_struct = upa

    if tr.is_complex(ant_spc_rx) or tr.numel(ant_spc_rx) != rx_struct or tr.any(ant_spc_rx <= 0):
        print('WARNING!!! input ant_spc_rx must be real, positive, and having ', rx_struct, ' element(s)')
        print('your input ant_spc_rx = ', ant_spc_rx)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(ant_spc_tx) or tr.numel(ant_spc_tx) != tx_struct or tr.any(ant_spc_tx <= 0):
        print('WARNING!!! input ant_spc_tx must be real, positive, and having ', tx_struct, ' element(s)')
        print('your input ant_spc_tx = ', ant_spc_tx)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(AoA) or tr.numel(AoA) != rx_struct or tr.any(AoA < -tr.pi) or tr.any(AoA > tr.pi):
        print('WARNING!!! input AoA must be real, having ', rx_struct, ' element(s), and ranging from -pi to pi')
        print('your input AoA = ', AoA)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(AoD) or tr.numel(AoD) != tx_struct or tr.any(AoD < -tr.pi) or tr.any(AoD > tr.pi):
        print('WARNING!!! input AoD must be real, having ', tx_struct, ' element(s), and ranging from -pi to pi')
        print('your input AoD = ', AoD)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(wave_len) or tr.numel(wave_len) != 1 or tr.any(wave_len <= 0):
        print('WARNING!!! input wave_len must be real, scalar, and positive')
        print('your input wave_len = ', wave_len)
        raise ValueError('INPUT ERROR')

    # Receiver Structure
    if rx_struct == ula:
        rx_response = tr.exp(-1j * 2 * tr.pi / wave_len * ant_spc_rx * tr.sin(AoA) * tr.arange(rx_size.item()))
        rx_response = rx_response.view(-1, 1)
    else:  # upa
        rx_response_v = tr.exp(-1j * 2 * tr.pi / wave_len * ant_spc_rx[0] * tr.sin(AoA[0]) * tr.arange(rx_size[0].item()))
        rx_response_v = rx_response_v.view(-1, 1)
        rx_response_h = tr.exp(-1j * 2 * tr.pi / wave_len * ant_spc_rx[1] * tr.cos(AoA[0]) * tr.sin(AoA[1])
                               * tr.arange(rx_size[1].item()))
        rx_response_h = rx_response_h.view(-1, 1)
        rx_response = tr.kron(rx_response_v, rx_response_h)

    # Transmitter Structure
    if tx_struct == ula:
        tx_response = tr.exp(-1j * 2 * tr.pi / wave_len * ant_spc_tx * tr.sin(AoD) * tr.arange(tx_size.item()))
        tx_response = tx_response.view(1, -1)
    else:  # upa
        tx_response_v = tr.exp(-1j * 2 * tr.pi / wave_len * ant_spc_tx[0] * tr.sin(AoD[0]) * tr.arange(tx_size[0].item()))
        tx_response_v = tx_response_v.view(1, -1)
        tx_response_h = tr.exp(-1j * 2 * tr.pi / wave_len * ant_spc_tx[1] * tr.cos(AoD[0]) * tr.sin(AoD[1])
                               * tr.arange(tx_size[1].item()))
        tx_response_h = tx_response_h.view(1, -1)
        tx_response = tr.kron(tx_response_v, tx_response_h)

    # Combine Receiver and Transmitter Structure
    rx_tx_response = tr.mm(rx_response, tx_response)
    return rx_tx_response

# ----------- EXAMPLES ----------- #
def ex_ula_ula():
    rx_size = tr.tensor([2])
    tx_size = tr.tensor([3])
    freq = 2.4e9
    wave_len = tr.tensor([3e8/freq])
    ant_spc_rx = ant_spc_tx = wave_len/2
    AoA = tr.tensor([2*(tr.rand(1)-0.5)])
    # AoA = tr.tensor([tr.pi / 6])
    AoD = tr.tensor([2 * (tr.rand(1) - 0.5)])
    # AoD = tr.tensor([tr.pi/4])
    LOS_component_smpl = LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len)
    print('LOS_component_smpl (ula-ula) = ', LOS_component_smpl)
def ex_ula_upa():
    rx_size = tr.tensor([2, 2])
    tx_size = tr.tensor([3])
    freq = 2.4e9
    wave_len = tr.tensor([3e8 / freq])  # ; print('wave_len = ', wave_len)
    ant_spc_rx = tr.tensor([wave_len / 2, wave_len / 2])
    ant_spc_tx = wave_len/2
    # AoA = tr.tensor([tr.pi / 8, tr.pi / 4])
    AoA = 2 * tr.pi * (tr.rand(2) - 0.5)
    # AoD = tr.tensor([tr.pi/4])
    AoD = 2 * tr.pi * (tr.rand(1) - 0.5)
    LOS_component_smpl = LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len)
    print('LOS_component_smpl (ula-upa) = ', LOS_component_smpl)
def ex_upa_upa():
    rx_size = tr.tensor([3, 4])
    tx_size = tr.tensor([2, 2])
    freq = 2.4e9
    wave_len = tr.tensor([3e8 / freq])
    ant_spc_rx = tr.tensor([wave_len / 2, wave_len / 2])
    ant_spc_tx = tr.tensor([wave_len / 2, wave_len / 2])
    AoA = 2 * tr.pi * (tr.rand(2) - 0.5)
    AoD = 2 * tr.pi * (tr.rand(2) - 0.5)
    LOS_component_smpl = LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len)
    print('LOS_component_smpl (upa-upa) = ', LOS_component_smpl)


ex_ula_ula()
ex_ula_upa()
ex_upa_upa()
