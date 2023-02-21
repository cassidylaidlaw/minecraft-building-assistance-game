"""
Code to interface with Project Malmo.
"""

import atexit
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple, TypedDict

import MalmoPython
import numpy as np
from typing_extensions import Literal

from .blocks import MinecraftBlocks
from .types import BlockLocation

if TYPE_CHECKING:
    from .mbag_env import MbagConfigDict

logger = logging.getLogger(__name__)


class MalmoEvent(TypedDict, total=False):
    command: str
    pressed: bool


class MalmoRayObservation(TypedDict):
    hitType: Literal["block", "entity", "item"]  # noqa: N815
    x: float
    y: float
    z: float
    inRange: bool  # noqa: N815


class MalmoObservationDict(TypedDict, total=False):
    world: List[str]
    goal: List[str]
    XPos: float
    YPos: float
    ZPos: float
    events: List[MalmoEvent]
    LineOfSight: MalmoRayObservation


class MalmoClient(object):
    ACTION_DELAY = 0.3

    agent_hosts: List[MalmoPython.AgentHost]
    experiment_id: str
    record_fname: Optional[str]
    ssh_processes: List[subprocess.Popen]

    def __init__(self):
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool_size = 0
        self.record_fname = None

        self.ssh_processes = []
        atexit.register(self._cleanup_ssh_processes)

    def get_player_name(self, player_index: int, env_config: "MbagConfigDict") -> str:
        player_name = env_config["players"][player_index].get("player_name")
        if player_name is None:
            player_name = f"player_{player_index}"
        # Player names cannot be longer than 16 character in Minecraft.
        return player_name[:16]

    def _get_agent_section_xml(self, player_index: int, env_config: "MbagConfigDict"):
        width, height, depth = env_config["world_size"]

        inventory_item_tags: List[str] = []
        if env_config["abilities"]["inf_blocks"]:
            for block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
                block_name = MinecraftBlocks.ID2NAME[block_id]
                inventory_item_tags.append(
                    f"""
                    <InventoryItem slot="{block_id}" type="{block_name}" />
                    """
                )
        inventory_items_xml = "\n".join(inventory_item_tags)

        command_handlers = """
        <AbsoluteMovementCommands />
        <DiscreteMovementCommands>
            <ModifierList type="deny-list">
                <command>jump</command>
            </ModifierList>
        </DiscreteMovementCommands>
        <InventoryCommands />
        <HumanLevelCommands>
            <ModifierList type="allow-list">
                <command>jump</command>
            </ModifierList>
        </HumanLevelCommands>
        <ChatCommands />
        <MissionQuitCommands />
        """

        observation_handlers = f"""
        <ObservationFromGrid>
            <Grid name="world" absoluteCoords="true">
                <min x="0" y="0" z="0" />
                <max x="{width - 1}" y="{height - 1}" z="{depth - 1}" />
            </Grid>
            <Grid name="goal" absoluteCoords="true">
                <min x="{width + 1}" y="0" z="0" />
                <max x="{width * 2}" y="{height - 1}" z="{depth - 1}" />
            </Grid>
        </ObservationFromGrid>
        <ObservationFromFullInventory />
        <ObservationFromFullStats />
        """

        player_config = env_config["players"][player_index]

        if player_config["is_human"]:
            observation_handlers += f"""
            <ObservationFromChat />
            <ObservationFromRecentCommands />
            <ObservationFromRay />
            <ObservationFromHuman />
            <ObservationFromSystem />
            """
        
        if player_config["is_minerl_agent"]:
            observation_handlers = """
            <VideoProducer want_depth="false">
                <Width>640</Width>
                <Height>360</Height>
            </VideoProducer>
            <ObservationFromFullInventory flat="false"/>
            <ObservationFromFullStats />
            """
            command_handlers = """
            <HumanLevelCommands />
            <AbsoluteMovementCommands />
            <ChatCommands />
            """

        return f"""
        <AgentSection mode="Creative">
            <Name>{self.get_player_name(player_index, env_config)}</Name>
            <AgentStart>
                <!-- <Placement x="{0.5 + player_index}" y="2" z="0.5" yaw="270"/> -->
                <Inventory>
                    {inventory_items_xml}
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                {observation_handlers}
                {command_handlers}
            </AgentHandlers>
        </AgentSection>
        """

    def _get_spectator_position(self, env_config: "MbagConfigDict") -> BlockLocation:
        width, height, depth = env_config["world_size"]
        x = width
        y = height // 2 + 1
        z = -width
        return x, y, z

    def _get_spectator_agent_section_xml(self, env_config: "MbagConfigDict") -> str:
        width, height, depth = env_config["world_size"]
        x, y, z = self._get_spectator_position(env_config)
        pitch = np.rad2deg(np.arctan((y - height / 2) / (depth / 2 - z)))
        return f"""
        <AgentSection mode="Creative">
            <Name>spectator</Name>
            <AgentStart>
                <Placement x="{x + 0.5}" y="{y}" z="{z + 0.5}" yaw="0" pitch="{pitch}" />
            </AgentStart>
            <AgentHandlers>
                <VideoProducer>
                    <Width>640</Width>
                    <Height>480</Height>
                </VideoProducer>
            </AgentHandlers>
        </AgentSection>
        """

    def _get_spectator_platform_drawing_decorator_xml(
        self, env_config: "MbagConfigDict"
    ) -> str:
        if env_config["malmo"]["use_spectator"]:
            x, y, z = self._get_spectator_position(env_config)
            return f"""
            <DrawCuboid
                type="bedrock"
                x1="{x}"
                y1="1"
                z1="{z}"
                x2="{x}"
                y2="{y - 1}"
                z2="{z}"
            />
            """
        else:
            return ""

    def _blocks_to_drawing_decorator_xml(
        self, blocks: MinecraftBlocks, offset: BlockLocation = (0, 0, 0)
    ) -> str:
        draw_tags: List[str] = []
        for (x, y, z), block_id in np.ndenumerate(blocks.blocks):
            block_name = MinecraftBlocks.ID2NAME[block_id]
            if (
                block_name == "air"
                or block_name == "bedrock"
                and y == 0
                or block_name == "dirt"
                and y == 1
            ):
                continue

            block_state = blocks.block_states[x, y, z]  # noqa, TODO: use this
            draw_tags.append(
                f"""
                <DrawBlock
                    type="{block_name}"
                    x="{x + offset[0]}"
                    y="{y + offset[1]}"
                    z="{z + offset[2]}"
                />
                """
            )
        return "\n".join(draw_tags)

    @staticmethod
    def _draw_wall(
        env_config: "MbagConfigDict",
        block_type: str,
        coord_1: Tuple[int, int, int],
        coord_2: Tuple[int, int, int],
    ) -> str:
        if env_config["malmo"]["restrict_players"]:
            return f"""
                <DrawCuboid
                    type="{block_type}"
                    x1="{coord_1[0]}"
                    y1="{coord_1[1]}"
                    z1="{coord_1[2]}"
                    x2="{coord_2[0]}"
                    y2="{coord_2[1]}"
                    z2="{coord_2[2]}"
                />
            """

        return ""

    def _get_mission_spec_xml(
        self,
        env_config: "MbagConfigDict",
        current_blocks: MinecraftBlocks,
        goal_blocks: MinecraftBlocks,
        force_reset: bool = True,
    ) -> str:
        width, height, depth = env_config["world_size"]
        force_reset_str = "true" if force_reset else "false"

        agent_section_xmls = [
            self._get_agent_section_xml(player_index, env_config)
            for player_index in range(env_config["num_players"])
        ]
        if env_config["malmo"]["use_spectator"]:
            agent_section_xmls.append(self._get_spectator_agent_section_xml(env_config))
        agent_sections_xml = "\n".join(agent_section_xmls)

        return f"""
        <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

            <About>
                <Summary>test</Summary>
            </About>

            <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>1000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                    <DefaultWorldGenerator />
                </ServerHandlers>
            </ServerSection>

            {agent_sections_xml}
        </Mission>
        """

        return f"""
        <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

            <About>
                <Summary>Minecraft Building Assistance Game</Summary>
            </About>

            <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>1000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                </ServerInitialConditions>
                <ServerHandlers>
                    <FlatWorldGenerator
                        forceReset="{force_reset_str}"
                        generatorString="3;1*minecraft:bedrock,1*minecraft:grass;minecraft:plains;"
                        destroyAfterUse="true"
                    />
                    <DrawingDecorator>
                        {self._draw_wall(env_config, "bedrock", (width, 2, -1), (width, height, depth))}
                        {self._draw_wall(env_config, "barrier", (-1, 2, -1), (-1, height, depth))}
                        {self._draw_wall(env_config, "barrier", (-1, 2, -1), (width, height, -1))}
                        {self._draw_wall(env_config, "barrier", (-1, 2, depth), (width, height, depth))}
                        {self._draw_wall(env_config, "barrier", (-1, height, -1), (width, height, depth))}
                        {self._blocks_to_drawing_decorator_xml(goal_blocks, (width+1, 0, 0))}
                        {self._blocks_to_drawing_decorator_xml(current_blocks)}
                        {self._get_spectator_platform_drawing_decorator_xml(env_config)}
                    </DrawingDecorator>
                    <BuildBattleDecorator>
                        <PlayerStructureBounds>
                            <min x="0" y="0" z="0" />
                            <max x="{width - 1}" y="{height - 1}" z="{depth - 1}" />
                        </PlayerStructureBounds>
                        <GoalStructureBounds>
                            <min x="{width + 1}" y="0" z="0" />
                            <max x="{width * 2}" y="{height - 1}" z="{depth - 1}" />
                        </GoalStructureBounds>
                    </BuildBattleDecorator>
                </ServerHandlers>
            </ServerSection>

            {agent_sections_xml}
        </Mission>
        """

    def _expand_client_pool(self, num_clients):
        while self.client_pool_size < num_clients:
            self.client_pool.add(
                MalmoPython.ClientInfo("127.0.0.1", self.client_pool_size + 10000)
            )
            self.client_pool_size += 1

    # This method based on code from multi_agent_test.py in the Project Malmo examples.
    def _safe_start_mission(
        self,
        agent_host: MalmoPython.AgentHost,
        mission: MalmoPython.MissionSpec,
        mission_record: MalmoPython.MissionRecordSpec,
        player_index: int,
        *,
        max_attempts: int = 1 if "pytest" in sys.modules else 5,
        ssh_args: Optional[List[str]] = None,
    ):
        used_attempts = 0
        logger.info(f"starting Malmo mission for player {player_index}")
        while True:
            try:
                # Attempt start:
                agent_host.startMission(
                    mission,
                    self.client_pool,
                    mission_record,
                    player_index,
                    self.experiment_id,
                )
                break
            except MalmoPython.MissionException as error:
                error_code = error.details.errorCode
                if error_code == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                    logger.info("server not quite ready yet, waiting...")
                    time.sleep(2)
                elif (
                    error_code
                    == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE
                ):
                    logger.warning("not enough available Minecraft instances running")
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        logger.info(
                            "will wait in case they are starting up; "
                            f"{max_attempts - used_attempts} attempts left",
                        )
                        time.sleep(2)
                elif (
                    error_code == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND
                ):
                    logger.warning(
                        "server not found; has the mission with role 0 been started yet?"
                    )
                    used_attempts += 1
                    if used_attempts < max_attempts:
                        logger.info(
                            "will wait and retry; "
                            f"{max_attempts - used_attempts} attempts left",
                        )
                        time.sleep(2)
                else:
                    raise error
                if used_attempts == max_attempts:
                    logger.error(
                        f"failed to start mission after {max_attempts} attempts"
                    )
                    raise error

        logger.info(f"Malmo mission successfully started for player {player_index}")

    def _try_get_minecraft_server_port(
        self,
        client_host: str,
        client_port: int,
    ) -> Optional[int]:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((client_host, client_port))
        client_socket.sendall(
            b"MALMO_FIND_SERVER" + self.experiment_id.encode("ascii") + b"\n"
        )

        message_len = 1024
        reply_with_header = b""
        while len(reply_with_header) < 4 + message_len:
            reply_with_header += client_socket.recv(1024)
            if len(reply_with_header) > 4:
                message_len = int.from_bytes(reply_with_header[:4], byteorder="big")

        reply = reply_with_header[4:]

        client_socket.close()

        if reply.startswith(b"MALMOS"):
            return int(reply.split(b":")[-1])
        else:
            return None

    def _open_ssh_tunnels(
        self, ssh_args: List[str], ports_to_forward: List[Tuple[str, int]]
    ):
        # Split across multiple SSH processes since it seems like SSH can't handle many more
        # than about 500 forwarded ports.
        while ports_to_forward:
            ports_to_forward_section = ports_to_forward[:500]
            ports_to_forward = ports_to_forward[500:]

            ssh_command = ["ssh", "-T"] + ssh_args
            for flag, port in ports_to_forward_section:
                ssh_command.extend([flag, f"{port}:localhost:{port}"])
            logger.info(
                f"opening SSH tunnels with command: {' '.join(ssh_command)[:200]}..."
            )
            ssh_process = subprocess.Popen(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )
            self.ssh_processes.append(ssh_process)

    def _cleanup_ssh_processes(self):
        while len(self.ssh_processes) > 0:
            ssh_process = self.ssh_processes.pop()
            ssh_process.terminate()
            if ssh_process.poll() is None:
                time.sleep(1)
            if ssh_process.poll() is None:
                ssh_process.kill()

    # This method based on code from multi_agent_test.py in the Project Malmo examples.
    def _safe_wait_for_start(self, agent_hosts: List[MalmoPython.AgentHost]):
        logger.info("waiting for the mission to start")
        agent_hosts_started = [False for _ in agent_hosts]
        start_time = time.time()
        time_out = 120  # Allow a two minute timeout.
        while not all(agent_hosts_started) and time.time() - start_time < time_out:
            states = [agent_host.peekWorldState() for agent_host in agent_hosts]
            agent_hosts_started = [state.has_mission_begun for state in states]
            errors = [error for state in states for error in state.errors]
            if len(errors) > 0:
                logger.error("errors waiting for mission start:")
                for e in errors:
                    logger.error(e.text)
                raise errors[0]
            time.sleep(0.1)
        if time.time() - start_time >= time_out:
            logger.error("timed out while waiting for mission to start")
            raise RuntimeError("timed out while waiting for mission to start")

    def _get_num_agents(self, env_config: "MbagConfigDict"):
        num_agents = env_config["num_players"]
        if env_config["malmo"]["use_spectator"]:
            num_agents += 1
        return num_agents

    def _get_spectator_agent_index(self, env_config: "MbagConfigDict") -> Optional[int]:
        if env_config["malmo"]["use_spectator"]:
            return env_config["num_players"]
        else:
            return None

    def _get_player_ssh_args(
        self, env_config: "MbagConfigDict", player_index: int
    ) -> Optional[List[str]]:
        all_ssh_args = env_config["malmo"].get("ssh_args")
        player_ssh_args: Optional[List[str]] = None
        if all_ssh_args is not None:
            player_ssh_args = all_ssh_args[player_index]
        if player_index == 0 and player_ssh_args is not None:
            raise ValueError("player 0 must be running locally")
        return player_ssh_args

    def _generate_record_fname(self, env_config: "MbagConfigDict"):
        video_dir = env_config["malmo"]["video_dir"]
        assert video_dir is not None
        video_index = 0
        while True:
            self.record_fname = os.path.join(video_dir, f"{video_index:06d}.tar.gz")
            if not os.path.exists(self.record_fname):
                return
            video_index += 1

    def start_mission(
        self,
        env_config: "MbagConfigDict",
        current_blocks: MinecraftBlocks,
        goal_blocks: MinecraftBlocks,
    ):
        # Set up SSH forwarding.
        for player_index in range(env_config["num_players"]):
            player_ssh_args = self._get_player_ssh_args(env_config, player_index)
            if player_ssh_args is not None:
                ports_to_forward: List[Tuple[str, int]] = []
                ports_to_forward.append(("-L", 10000 + player_index))
                for port in range(10000 + self._get_num_agents(env_config), 11001):
                    ports_to_forward.append(("-R", port))
                self._open_ssh_tunnels(player_ssh_args, ports_to_forward)

        self._expand_client_pool(self._get_num_agents(env_config))
        self.experiment_id = str(uuid.uuid4())
        self.record_fname = None
        minecraft_server_port: Optional[int] = None

        self.agent_hosts = []
        for player_index in range(self._get_num_agents(env_config)):
            agent_host = MalmoPython.AgentHost()
            agent_host.setObservationsPolicy(
                MalmoPython.ObservationsPolicy.KEEP_ALL_OBSERVATIONS
            )
            self.agent_hosts.append(agent_host)
            mission_spec_xml = self._get_mission_spec_xml(
                env_config, current_blocks, goal_blocks, force_reset=player_index == 0
            )
            record_spec = MalmoPython.MissionRecordSpec()
            if player_index == self._get_spectator_agent_index(env_config):
                if env_config["malmo"]["video_dir"]:
                    self._generate_record_fname(env_config)
                    record_spec = MalmoPython.MissionRecordSpec(self.record_fname)
                    record_spec.recordMP4(
                        MalmoPython.FrameType.VIDEO, 20, 400000, False
                    )

            # Open up another tunnel for the Minecraft server once the first player's
            # game is started.
            player_ssh_args = self._get_player_ssh_args(env_config, player_index)
            if player_ssh_args is not None:
                assert minecraft_server_port is not None
                self._open_ssh_tunnels(player_ssh_args, [("-R", minecraft_server_port)])
                # Give some time for SSH to start.
                time.sleep(10)

            self._safe_start_mission(
                agent_host,
                MalmoPython.MissionSpec(mission_spec_xml, True),
                record_spec,
                player_index,
                ssh_args=player_ssh_args,
            )
            if player_index == 0 and self._get_num_agents(env_config) > 1:
                # Wait for the Minecraft server to start and get its port.
                timeout_seconds_left = 60
                logger.info("waiting for the Minecraft server to start...")
                while minecraft_server_port is None:
                    time.sleep(2)
                    timeout_seconds_left -= 2
                    minecraft_server_port = self._try_get_minecraft_server_port(
                        "localhost",
                        10000,
                    )
                    if timeout_seconds_left <= 0:
                        break

                if minecraft_server_port is None:
                    raise RuntimeError(
                        "timed out waiting for Minecraft server to start"
                    )

        self._safe_wait_for_start(self.agent_hosts)

    def send_command(self, player_index: int, command: str):
        logger.debug(f"player {player_index} command: {command}")
        self.agent_hosts[player_index].sendCommand(command)

    def get_observations(
        self, player_index: int
    ) -> List[Tuple[datetime, MalmoObservationDict]]:
        agent_host = self.agent_hosts[player_index]
        world_state = agent_host.getWorldState()
        if not world_state.is_mission_running:
            return []
        else:
            observation_tuples: List[Tuple[datetime, MalmoObservationDict]] = []
            for observation in world_state.observations:
                observation_tuples.append(
                    (
                        observation.timestamp,
                        json.loads(observation.text),
                    )
                )
            return observation_tuples

    def get_observation(self, player_index: int) -> Optional[MalmoObservationDict]:
        observations = self.get_observations(player_index)
        if len(observations) > 0:
            timestamp, observation = observations[0]
            return observation
        else:
            return None

    def end_mission(self):
        for player_index in range(len(self.agent_hosts)):
            self.send_command(player_index, "quit")

        # Important to get rid of agent hosts, which triggers video writing for some
        # reason.
        self.agent_hosts = []

        self._save_specatator_video()

        self._cleanup_ssh_processes()

    def _save_specatator_video(self):
        if self.record_fname is None:
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            record_tar = tarfile.open(self.record_fname, "r:gz")
            video_member_name = None
            for member_name in record_tar.getnames():
                if member_name.endswith("/video.mp4"):
                    video_member_name = member_name
                    break
            assert video_member_name is not None
            record_tar.extract(video_member_name, temp_dir)
            shutil.move(
                os.path.join(temp_dir, video_member_name),
                self.record_fname[: -len(".tar.gz")] + ".mp4",
            )
