import { SendCommandRequest, StreamRobotStateRequest } from "./generated/continuonbrain_link";
import { Episode, RobotState } from "./generated/rlds_episode";

export const exampleRequest: StreamRobotStateRequest = { clientId: "xr-dev" };

export function buildIdleState(): RobotState {
  return {
    timestampNanos: 0n,
    jointPositions: [],
    endEffectorPose: undefined,
    gripperOpen: false,
    frameId: "base", 
    jointVelocities: [],
    jointEfforts: [],
    endEffectorTwist: [],
    wallTimeMillis: 0n
  };
}

export function buildAckedCommand(): SendCommandRequest {
  return {
    clientId: "xr-dev",
    controlMode: 1,
    targetFrequencyHz: 30,
    safety: undefined,
    eeVelocity: {
      linearMps: { x: 0, y: 0, z: 0 },
      angularRadS: { x: 0, y: 0, z: 0 },
      referenceFrame: 1
    }
  };
}

export function emptyEpisode(): Episode {
  return {
    metadata: {
      xrMode: "trainer",
      controlRole: "human_teleop",
      environmentId: "lab-mock",
      tags: ["smoke"],
      software: undefined
    },
    steps: []
  };
}
