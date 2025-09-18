from __future__ import annotations
from typing import Dict, Tuple
from dynamixel_sdk import *  # type: ignore

# X/XL/XM 시리즈(Protocol 2.0) 기본 주소
ADDR_TORQUE_ENABLE     = 64
ADDR_OPERATING_MODE    = 11
ADDR_GOAL_POSITION     = 116
ADDR_PRESENT_POSITION  = 132
LEN_GOAL_POSITION      = 4

OP_MODE_CURRENT        = 0
OP_MODE_VELOCITY       = 1
OP_MODE_POSITION       = 3
OP_MODE_CURRENT_POS    = 5

class DynamixelBus:
    def __init__(self, port_name: str, baudrate: int = 1000000, protocol: float = 2.0, log=print):
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(protocol)
        self.log = log
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port {port_name}")
        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}")

    def ping(self, dxl_id: int) -> Tuple[int, int]:
        dxl_model_number, rc, err = self.packet.ping(self.port, dxl_id)
        if rc != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(rc))
        if err != 0:
            self.log(f"[WARN] ping id={dxl_id}: {self.packet.getRxPacketError(err)}")
        return dxl_model_number, err

    def write1(self, dxl_id: int, addr: int, value: int):
        rc, err = self.packet.write1ByteTxRx(self.port, dxl_id, addr, value)
        if rc != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(rc))
        if err:
            self.log(f"[WARN] write1 id={dxl_id}: {self.packet.getRxPacketError(err)}")

    def write4(self, dxl_id: int, addr: int, value: int):
        rc, err = self.packet.write4ByteTxRx(self.port, dxl_id, addr, value)
        if rc != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(rc))
        if err:
            self.log(f"[WARN] write4 id={dxl_id}: {self.packet.getRxPacketError(err)}")

    def read4(self, dxl_id: int, addr: int) -> int:
        val, rc, err = self.packet.read4ByteTxRx(self.port, dxl_id, addr)
        if rc != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(rc))
        if err:
            self.log(f"[WARN] read4 id={dxl_id}: {self.packet.getRxPacketError(err)}")
        return val

    def set_operating_mode(self, dxl_id: int, mode: int):
        # Torque off → 모드 변경 → 필요시 다시 on
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, 0)
        self.write1(dxl_id, ADDR_OPERATING_MODE, mode)

    def torque_enable(self, dxl_id: int, enable: bool):
        self.write1(dxl_id, ADDR_TORQUE_ENABLE, 1 if enable else 0)

    def sync_write_goal_positions(self, id_to_pos: Dict[int, int]):
        group = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for dxl_id, pos in id_to_pos.items():
            param = [pos & 0xFF, (pos >> 8) & 0xFF, (pos >> 16) & 0xFF, (pos >> 24) & 0xFF]
            if not group.addParam(dxl_id, bytes(param)):
                raise RuntimeError(f"GroupSyncWrite addParam failed id={dxl_id}")
        rc = group.txPacket()
        if rc != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(rc))
        group.clearParam()

    def close(self):
        try:
            self.port.closePort()
        except Exception:
            pass
