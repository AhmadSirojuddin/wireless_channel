"""
Generate a random wireless communication channel with various models, including Rayleigh/Rician fading, narrow/wide band.
This library uses the torch framework, so almost all data is the torch's tensor object.
"""

import torch as tr

pi = tr.pi

def dB_to_lin(input):
    return 10 ** (input / 10)

def uWave_ch(ch_size, LOS_component=tr.tensor([0]), Rician_fact=tr.tensor([0]), dist=tr.tensor([1]),
             dist_ref=tr.tensor([1]), path_loss_ref=tr.tensor([1]), path_loss_exp=tr.tensor([0]),
             path_num=tr.tensor([1]), multipath_decay=tr.tensor([1])):
    """
    Generates a wideband channel impulse response for MIMO channel microwave system using Rician model.
    All inputs must be the torch's tensor object.

    Inputs:
    A. Channel size
        (1) ch_size : 1D tensor with numel=2 representing the number of Rx and Tx antenna.
                      Num Rx antenna->0th element, num Tx antenna->1st element.

    B. Rician Model
        (2) LOS_component : Two dimension tensor of size ch_size representing the LoS channel component.
                            All elements must have absolute value = 1
        (3) Rician_fact : Rician Factor, in linear scale (not in dB).
                          Convert it to linear scale first if the original value is in dB.
                          'Rician_fact' must be real, scalar, and non-negative.
                          'Rician_fact=0' means Rayleigh fading channel is considered.
        Note : If Rayleigh channel is considered, inputs (2) and (3) can be ignored.

    C. PathLoss
        (4) dist : Distance from Tx to Rx.
                   It can either a scalar or matrix of size 'ch_size' representing the distances of each Tx-Rx antenna pairs.
                   All input elements must be real and positive.
        (5) dist_ref : Reference distance at which 'path_loss_ref' is defined.
                       'dist_ref' must be real, scalar, and positive.
        (6) path_loss_ref : Path loss at 'dist_ref', in linear scale (not in dB).
                            'path_loss_ref' must be real, scalar, and positive.
        (7) path_loss_exp : Path loss exponent, must be real, scalar,and non-negative
        Note : If normalized-distance channel model is considered, inputs (4)-(7) can be ignored

    D. Multipath Channel
        (8) path_num : number of channel impulse response for all channel.
                       Can be a scalar or a matrix.
                       All input elements must be real, integer, and positive
        (9) multipath_decay : Decay factor for the multi path channel impulse response model.
                              The underlying model can be found in the paper entitled:
                              'The Effects of Time Delay Spread on Portable Radio Communications Channels
                              with Digital Modulation'.
                              This input must be real, scalar, and positive.
        Note : If narrow-band channel is considered, inputs (8)-(9) can be ignored.

    Output:
    A. For narrow band case (path_num=1), output is a tensor of size [Rx, Tx].
    B. For wide band case (path_num>1), output is a tensor of size  [max(path_num), Rx, Tx]
    """

    # The following code part is for a prevention from wrong input. You can comment them once you are sure with your input
    incr = 10 ** -3  # Just small tolerance
    if tr.is_complex(ch_size) or tr.numel(ch_size) != 2 or tr.any(ch_size < 1) or tr.is_floating_point(ch_size):
        print('WARNING!!! input ch_size must be real, integer, positive, and a vector with two elements')
        print('your input ch_size = ', ch_size)
        raise ValueError('INPUT ERROR')

    if Rician_fact == 0:
        LOS_component = tr.zeros(ch_size.tolist())  # dummy
    else:
        if not tr.equal(ch_size, tr.tensor(LOS_component.size())) or tr.any(
              tr.reshape(LOS_component, (-1,)).abs() < 1 - incr) \
              or tr.any(tr.reshape(LOS_component, (-1,)).abs() > 1 + incr):
            print(
                'WARNING!!! input LOS_component must be a size of ch_size, and all absolute value of element must be one')
            print('your input LOS_component = ', LOS_component)
            print('LOS_component size = ', LOS_component.size())
            print('absolute value of LOS_component = ', tr.abs(LOS_component))
            raise ValueError('INPUT ERROR')
    if tr.is_complex(Rician_fact) or tr.numel(Rician_fact) != 1 or Rician_fact < 0:
        print('WARNING!!! input Rician_fact must be real scalar non-negative')
        print('your input Rician_fact = ', Rician_fact)
        raise ValueError('INPUT ERROR')

    if tr.is_complex(dist) or (tr.numel(dist) != 1 and not tr.equal(ch_size, tr.tensor(dist.size()))) \
          or tr.any(tr.reshape(dist, (-1,)) <= 0):
        print(
            'WARNING!!! input dist must be real, positive, and a scalar or matrix whose size is the same with ch_size')
        print('your input dist = ', dist)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(dist_ref) or tr.numel(dist_ref) != 1 or dist_ref <= 0:
        print('WARNING!!! input dist_ref must be a scalar, real, positive')
        print('your input dist_ref = ', dist_ref)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(path_loss_ref) or tr.numel(path_loss_ref) != 1 or path_loss_ref <= 0:
        print('WARNING!!! input path_loss_ref must be a scalar, real, positive')
        print('your input path_loss_ref = ', path_loss_ref)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(path_loss_exp) or tr.numel(path_loss_exp) != 1 or path_loss_exp < 0:
        print('WARNING!!! input path_loss_exp must be a scalar, real, positive')
        print('your input path_loss_exp = ', path_loss_exp)
        raise ValueError('INPUT ERROR')

    if tr.is_complex(path_num) or (tr.numel(path_num) != 1 and not tr.equal(ch_size, tr.tensor(path_num.size()))) \
          or tr.is_floating_point(path_num) or tr.any(tr.reshape(path_num, (-1,)) <= 0):
        print('WARNING!!! input path_num must be real, scalar, integer, and positive')
        print('your input path_num = ', path_num)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(multipath_decay) or tr.numel(multipath_decay) != 1 or multipath_decay <= 0:
        print('WARNING!!! input multipath_decay must be real, scalar, and positive')
        print('your input multipath_decay = ', multipath_decay)
        raise ValueError('INPUT ERROR')

    # The main code part begins here
    nPath = tr.reshape(path_num, (-1,)).max()  # ; print('nPath = ', nPath)
    if tr.numel(dist) == 1:
        dist = tr.ones(ch_size.tolist()) * dist  # ; print('dist = ', dist)
    if tr.numel(path_num):
        path_num = tr.ones(ch_size.tolist()) * path_num  # ; print('path_num = ', path_num)

    # ----------- First path channel ----------- #
    first_pow_profile = path_loss_ref * ((dist / dist_ref) ** -path_loss_exp)
    NLOS_component = (tr.randn(ch_size.tolist()) + 1j * tr.randn(ch_size.tolist())) / (
          2 ** 0.5)  # ; print('NLOS_component = ', NLOS_component)
    first_tap_ch = (first_pow_profile ** 0.5) * ((Rician_fact / (1 + Rician_fact)) ** 0.5 * LOS_component + (
          1 / (1 + Rician_fact)) ** 0.5 * NLOS_component)
    # print('first_tap_ch = ', first_tap_ch)

    if nPath == 1:
        return first_tap_ch
    else:
        # ----------- Multi path channel ----------- #
        ch = tr.zeros([nPath, ch_size[0].tolist(), ch_size[1].tolist()], dtype=tr.complex128)
        ch[0, :, :] = first_tap_ch
        for ch_th in range(1, nPath):
            # print('------- ch_th = ', ch_th, ' -------')
            pow_profile = first_pow_profile * tr.exp(-ch_th / multipath_decay)  # ; print('pow_profile = ', pow_profile)
            pow_profile = pow_profile * (path_num - ch_th > 0)  # ; print('pow_profile = ', pow_profile)
            ch[ch_th, :, :] = (pow_profile ** 0.5) * (tr.randn(ch_size.tolist()) + 1j * tr.randn(ch_size.tolist())) / (
                  2 ** 0.5)
        return ch

def LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len):
    """
    Creates a LOS component for the wireless MIMO channel. Either ULA (Uniform Linear Array) or UPA
    (Uniform Planar Array) structures can apply to the receiver or the transmitter. If ULA is considered at the receiver,
    the variables regarding Rx (rx_size, ant_spc_rx, AoA) are scalar. If UPA is considered, those variables are
    2D-tensors of size 2. The same scenario applies to the transmitter.

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
    if tr.is_complex(AoA) or tr.numel(AoA) != rx_struct or tr.any(AoA < -pi) or tr.any(AoA > pi):
        print('WARNING!!! input AoA must be real, having ', rx_struct, ' element(s), and ranging from -pi to pi')
        print('your input AoA = ', AoA)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(AoD) or tr.numel(AoD) != tx_struct or tr.any(AoD < -pi) or tr.any(AoD > pi):
        print('WARNING!!! input AoD must be real, having ', tx_struct, ' element(s), and ranging from -pi to pi')
        print('your input AoD = ', AoD)
        raise ValueError('INPUT ERROR')
    if tr.is_complex(wave_len) or tr.numel(wave_len) != 1 or tr.any(wave_len <= 0):
        print('WARNING!!! input wave_len must be real, scalar, and positive')
        print('your input wave_len = ', wave_len)
        raise ValueError('INPUT ERROR')

    # Receiver Structure
    if rx_struct == ula:
        rx_response = tr.exp(-1j * 2 * pi / wave_len * ant_spc_rx * tr.sin(AoA) * tr.arange(rx_size.item()))
        rx_response = rx_response.view(-1, 1)
    else:  # upa
        rx_response_v = tr.exp(-1j * 2 * pi / wave_len * ant_spc_rx[0] * tr.sin(AoA[0]) * tr.arange(rx_size[0].item()))
        rx_response_v = rx_response_v.view(-1, 1)
        rx_response_h = tr.exp(-1j * 2 * pi / wave_len * ant_spc_rx[1] * tr.cos(AoA[0]) * tr.sin(AoA[1])
                               * tr.arange(rx_size[1].item()))
        rx_response_h = rx_response_h.view(-1, 1)
        rx_response = tr.kron(rx_response_v, rx_response_h)

    # Transmitter Structure
    if tx_struct == ula:
        tx_response = tr.exp(-1j * 2 * pi / wave_len * ant_spc_tx * tr.sin(AoD) * tr.arange(tx_size.item()))
        tx_response = tx_response.view(1, -1)
    else:  # upa
        tx_response_v = tr.exp(-1j * 2 * pi / wave_len * ant_spc_tx[0] * tr.sin(AoD[0]) * tr.arange(tx_size[0].item()))
        tx_response_v = tx_response_v.view(1, -1)
        tx_response_h = tr.exp(-1j * 2 * pi / wave_len * ant_spc_tx[1] * tr.cos(AoD[0]) * tr.sin(AoD[1])
                               * tr.arange(tx_size[1].item()))
        tx_response_h = tx_response_h.view(1, -1)
        tx_response = tr.kron(tx_response_v, tx_response_h)

    # Combine Receiver and Transmitter Structure
    rx_tx_response = tr.mm(rx_response, tx_response)
    return rx_tx_response
