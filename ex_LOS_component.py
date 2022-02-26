import torch as tr
import wireless_channel as wc

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
    LOS_component = wc.LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len)
    print('LOS_component = ', LOS_component)
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
    LOS_component = wc.LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len)
    print('LOS_component = ', LOS_component)
def ex_upa_upa():
    rx_size = tr.tensor([3, 4])
    tx_size = tr.tensor([2, 2])
    freq = 2.4e9
    wave_len = tr.tensor([3e8 / freq])
    ant_spc_rx = tr.tensor([wave_len / 2, wave_len / 2])
    ant_spc_tx = tr.tensor([wave_len / 2, wave_len / 2])
    AoA = 2 * tr.pi * (tr.rand(2) - 0.5)
    AoD = 2 * tr.pi * (tr.rand(2) - 0.5)
    LOS_component = wc.LOS_component(rx_size, tx_size, ant_spc_rx, ant_spc_tx, AoA, AoD, wave_len)
    print('LOS_component = ', LOS_component)


ex_ula_ula()
ex_ula_upa()
ex_upa_upa()
