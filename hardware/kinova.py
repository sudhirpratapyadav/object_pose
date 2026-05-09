"""
Kinova Gen3 hardware interface.

Direct Kortex API communication -- TCP for high-level commands,
UDP for real-time torque control and feedback.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import (
    Session_pb2,
    Base_pb2,
    BaseCyclic_pb2,
    ActuatorConfig_pb2,
)
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient


NUM_JOINTS = 7
TCP_PORT = 10000
UDP_PORT = 10001
READY_STATES = [Base_pb2.ARMSTATE_SERVOING_READY, Base_pb2.ARMSTATE_SERVOING_MANUALLY_CONTROLLED]  # 7, 9


@dataclass
class RobotState:
    """Snapshot of robot feedback."""
    positions_deg: np.ndarray   # (7,)
    velocities_deg: np.ndarray  # (7,)
    torques: np.ndarray         # (7,) measured torques in Nm
    timestamp: float


class KinovaHardware:
    """
    Low-level Kinova Gen3 I/O.

    Handles connection lifecycle, mode switching, torque/position
    commands, and gripper control. No control logic.
    """

    def __init__(self, ip: str, username: str = "admin", password: str = "admin"):
        self.ip = ip
        self.username = username
        self.password = password

        self._tcp_transport = None
        self._udp_transport = None
        self._tcp_router = None
        self._udp_router = None
        self._tcp_session = None
        self._udp_session = None

        self.base = None
        self.base_cyclic = None
        self.actuator_config = None

        self._cmd = None
        self._send_opts = None
        self._in_torque_mode = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Establish TCP + UDP connections. Raises on failure."""
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 10000
        session_info.connection_inactivity_timeout = 2000

        # TCP
        self._tcp_transport = TCPTransport()
        self._tcp_router = RouterClient(
            self._tcp_transport, RouterClient.basicErrorCallback
        )
        self._tcp_transport.connect(self.ip, TCP_PORT)
        self._tcp_session = SessionManager(self._tcp_router)
        self._tcp_session.CreateSession(session_info)

        self.base = BaseClient(self._tcp_router)
        self.actuator_config = ActuatorConfigClient(self._tcp_router)

        # UDP
        self._udp_transport = UDPTransport()
        self._udp_router = RouterClient(
            self._udp_transport, RouterClient.basicErrorCallback
        )
        self._udp_transport.connect(self.ip, UDP_PORT)
        self._udp_session = SessionManager(self._udp_router)
        self._udp_session.CreateSession(session_info)

        self.base_cyclic = BaseCyclicClient(self._udp_router)

        self._init_command()

    def disconnect(self):
        """Close everything safely."""
        if self._in_torque_mode:
            try:
                self.set_torque_mode(False)
            except Exception:
                pass

        opts = RouterClientSendOptions()
        opts.timeout_ms = 1000

        for session in (self._udp_session, self._tcp_session):
            if session:
                try:
                    session.CloseSession(opts)
                except Exception:
                    pass

        for transport in (self._udp_transport, self._tcp_transport):
            if transport:
                try:
                    transport.disconnect()
                except Exception:
                    pass

        self.base = None
        self.base_cyclic = None
        self.actuator_config = None
        self._cmd = None
        self._in_torque_mode = False

    def _init_command(self):
        """Pre-allocate the reusable UDP command structure."""
        self._cmd = BaseCyclic_pb2.Command()
        for i in range(NUM_JOINTS):
            actuator = self._cmd.actuators.add()
            actuator.flags = 1
            actuator.command_id = 0
        self._cmd.frame_id = 0

        # Gripper via interconnect (used in low-level mode)
        self._cmd.interconnect.command_id.identifier = 0
        motor = self._cmd.interconnect.gripper_command.motor_cmd.add()
        motor.motor_id = 0
        motor.position = 0.0
        motor.velocity = 100.0

        self._send_opts = RouterClientSendOptions()
        self._send_opts.andForget = False
        self._send_opts.delay_ms = 0
        self._send_opts.timeout_ms = 10

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_servoing_mode(self, low_level: bool, timeout: float = 2.0) -> bool:
        """
        Switch between LOW_LEVEL_SERVOING and SINGLE_LEVEL_SERVOING.
        Blocks until mode is verified or timeout.
        Returns True if successful, False otherwise.
        """
        mode_info = Base_pb2.ServoingModeInformation()
        mode_info.servoing_mode = (
            Base_pb2.LOW_LEVEL_SERVOING if low_level
            else Base_pb2.SINGLE_LEVEL_SERVOING
        )
        self.base.SetServoingMode(mode_info)
        time.sleep(0.3)  # Give time for mode switch to initiate
        return self.verify_servoing_mode(expected_low_level=low_level, timeout=timeout)

    def set_torque_mode(self, enabled: bool):
        """Set all actuators to TORQUE or POSITION control mode."""
        mode_info = ActuatorConfig_pb2.ControlModeInformation()
        mode_info.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            'TORQUE' if enabled else 'POSITION'
        )
        for i in range(NUM_JOINTS):
            self.actuator_config.SetControlMode(mode_info, i + 1)
        self._in_torque_mode = enabled

    @property
    def in_torque_mode(self) -> bool:
        return self._in_torque_mode

    # ------------------------------------------------------------------
    # Arm state
    # ------------------------------------------------------------------

    def clear_faults(self):
        self.base.ClearFaults()

    def stop(self):
        try:
            self.base.Stop()
        except Exception:
            pass

    def is_ready(self) -> bool:
        try:
            return self.base.GetArmState().active_state in READY_STATES
        except Exception:
            return False

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Poll arm state, auto-clearing faults. Returns True if ready."""
        deadline = time.time() + timeout
        last_state = None
        while time.time() < deadline:
            if self.is_ready():
                logger.info("Arm is ready")
                return True
            try:
                arm_state = self.base.GetArmState()
                if arm_state.active_state != last_state:
                    logger.info(f"Arm state: {arm_state.active_state} (waiting for ready)")
                    last_state = arm_state.active_state
                if arm_state.active_state == Base_pb2.ARMSTATE_IN_FAULT:
                    logger.warning("Arm in fault state, attempting to clear...")
                    self.clear_faults()
                elif arm_state.active_state == Base_pb2.ARMSTATE_SERVOING_MANUALLY_CONTROLLED:
                    logger.warning("Arm manually controlled, stopping...")
                    self.stop()
                elif arm_state.active_state == Base_pb2.ARMSTATE_SERVOING_PRE_READY:
                    logger.info("Arm in PRE_READY, clearing faults…")
                    self.clear_faults()
            except Exception as e:
                logger.warning(f"Error checking arm state: {e}")
            time.sleep(0.1)
        logger.error(f"Arm not ready after {timeout}s timeout. Last state: {last_state}")
        return False

    def get_servoing_mode(self) -> str:
        """Get current servoing mode. Returns 'LOW_LEVEL' or 'SINGLE_LEVEL' or 'UNKNOWN'."""
        try:
            servoing_mode = self.base.GetServoingMode()
            if servoing_mode.servoing_mode == Base_pb2.LOW_LEVEL_SERVOING:
                return "LOW_LEVEL"
            elif servoing_mode.servoing_mode == Base_pb2.SINGLE_LEVEL_SERVOING:
                return "SINGLE_LEVEL"
            else:
                return f"UNKNOWN({servoing_mode.servoing_mode})"
        except Exception as e:
            logger.error(f"Error getting servoing mode: {e}")
            return "ERROR"

    def verify_servoing_mode(self, expected_low_level: bool, timeout: float = 2.0) -> bool:
        """Verify robot is in expected servoing mode. Returns True if correct."""
        expected = "LOW_LEVEL" if expected_low_level else "SINGLE_LEVEL"
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = self.get_servoing_mode()
            if current == expected:
                logger.info(f"Servoing mode verified: {expected}")
                return True
            logger.warning(f"Servoing mode mismatch: expected {expected}, got {current}")
            time.sleep(0.1)
        logger.error(f"Servoing mode verification failed after {timeout}s")
        return False

    # ------------------------------------------------------------------
    # Real-time I/O (UDP)
    # ------------------------------------------------------------------

    def read_state(self) -> RobotState:
        """Read current robot state via UDP. Raises on failure."""
        fb = self.base_cyclic.RefreshFeedback()
        return RobotState(
            positions_deg=np.array([fb.actuators[i].position for i in range(NUM_JOINTS)]),
            velocities_deg=np.array([fb.actuators[i].velocity for i in range(NUM_JOINTS)]),
            torques=np.array([fb.actuators[i].torque for i in range(NUM_JOINTS)]),
            timestamp=time.time(),
        )

    def send_torques(
        self,
        torques: np.ndarray,
        positions_deg: np.ndarray,
        gripper_position: float = 0.0,
    ) -> RobotState:
        """
        Send joint torques and return fresh feedback.

        Args:
            torques: (7,) joint torques in Nm.
            positions_deg: (7,) current positions echoed back (required by API).
            gripper_position: 0.0 = open, 1.0 = closed.
        """
        self._cmd.frame_id = (self._cmd.frame_id + 1) % 65536
        for i in range(NUM_JOINTS):
            self._cmd.actuators[i].position = positions_deg[i]
            self._cmd.actuators[i].torque_joint = torques[i]
            self._cmd.actuators[i].command_id = self._cmd.frame_id

        self._cmd.interconnect.command_id.identifier = self._cmd.frame_id
        self._cmd.interconnect.gripper_command.motor_cmd[0].position = (
            gripper_position * 100.0
        )

        fb = self.base_cyclic.Refresh(self._cmd, 0, self._send_opts)
        return RobotState(
            positions_deg=np.array([fb.actuators[i].position for i in range(NUM_JOINTS)]),
            velocities_deg=np.array([fb.actuators[i].velocity for i in range(NUM_JOINTS)]),
            torques=np.array([fb.actuators[i].torque for i in range(NUM_JOINTS)]),
            timestamp=time.time(),
        )

    def send_positions(
        self,
        positions_deg: np.ndarray,
        gripper_position: float = 0.0,
    ) -> RobotState:
        """Send position command (for stabilization before torque mode)."""
        return self.send_torques(np.zeros(NUM_JOINTS), positions_deg, gripper_position)

    # ------------------------------------------------------------------
    # High-level actions (TCP)
    # ------------------------------------------------------------------

    def go_to_joints(self, target_deg: np.ndarray, duration: float = 8.0) -> bool:
        """
        Blocking joint move using Kinova's built-in trajectory planner.

        Must be in SINGLE_LEVEL_SERVOING (high-level) mode.
        Returns True on success.
        """
        action = Base_pb2.Action()
        action.name = "JointMove"
        action.reach_joint_angles.constraint.type = Base_pb2.JOINT_CONSTRAINT_DURATION
        action.reach_joint_angles.constraint.value = duration

        for i, angle in enumerate(target_deg):
            joint = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint.joint_identifier = i
            joint.value = float(angle)

        done = [False]
        success = [False]
        abort_info = [None]

        def _on_action(notif):
            event_names = {
                Base_pb2.ACTION_START: "ACTION_START",
                Base_pb2.ACTION_END: "ACTION_END",
                Base_pb2.ACTION_ABORT: "ACTION_ABORT",
                Base_pb2.ACTION_PAUSE: "ACTION_PAUSE",
            }
            event_name = event_names.get(notif.action_event, f"UNKNOWN({notif.action_event})")
            logger.debug(f"Action notification: {event_name}")

            if notif.action_event == Base_pb2.ACTION_END:
                done[0] = True
                success[0] = True
            elif notif.action_event == Base_pb2.ACTION_ABORT:
                abort_info[0] = notif
                done[0] = True

        handle = self.base.OnNotificationActionTopic(
            _on_action, Base_pb2.NotificationOptions()
        )
        try:
            logger.info(f"Executing JointMove to {target_deg} (duration={duration}s)")
            self.base.ExecuteAction(action)
            deadline = time.time() + duration + 10.0
            while not done[0] and time.time() < deadline:
                time.sleep(0.1)

            if not done[0]:
                logger.error(f"JointMove timed out after {duration + 10.0}s")
            elif abort_info[0] is not None:
                logger.error(f"JointMove aborted. Notification: {abort_info[0]}")
                # Try to get current arm state for more context
                try:
                    arm_state = self.base.GetArmState()
                    logger.error(f"Arm state after abort: {arm_state}")
                except Exception as e:
                    logger.warning(f"Could not get arm state: {e}")
        finally:
            self.base.Unsubscribe(handle)

        return success[0]

    # ------------------------------------------------------------------
    # Gripper (TCP -- for use in high-level mode)
    # ------------------------------------------------------------------

    def send_gripper_command(self, position: float):
        """Send gripper position command. 0.0 = open, 1.0 = closed."""
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = max(0.0, min(1.0, position))
        self.base.SendGripperCommand(cmd)

    def read_gripper_position(self) -> float:
        """Read gripper position. 0.0 = open, 1.0 = closed."""
        try:
            req = Base_pb2.GripperRequest()
            req.mode = Base_pb2.GRIPPER_POSITION
            measure = self.base.GetMeasuredGripperMovement(req)
            if measure.finger:
                return float(measure.finger[0].value)
        except Exception:
            pass
        return 0.0
