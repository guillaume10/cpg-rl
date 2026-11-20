# License: see [LICENSE]

import torch 

def ComputeIK(legID, x, y, z, robot_name="GO1"):
    if robot_name == "GO1":
        l1 = 0.08
        l2 = 0.213
        l3 = 0.213
    elif robot_name == "GO2":
        l1 = 0.0955
        l2 = 0.213
        l3 = 0.213
    else:
        raise ValueError(f"******* incorrect robot_name: {robot_name} *******")

    # get_domain
    D = (y**2 + (-z)**2 - l1**2 +
        (-x)**2 - l2**2 - l3**2) / (
                 2 * l3 * l2)
    D = torch.clip(D, -1.0, 1.0)

    # check Right vs Left leg for hip angle
    sideSign = 1
    if legID == 0 or legID == 2:
      sideSign = -1

    wrist_angle = torch.atan2(-torch.sqrt(1 - D**2), D)
    sqrt_component = y**2 + (-z)**2 - l1**2
    sqrt_component = torch.clamp(sqrt_component,min=0)
    shoulder_angle = -torch.atan2(z, y) - torch.atan2(
        torch.sqrt(sqrt_component), sideSign*l1*torch.ones_like(x))
    elbow_angle = torch.atan2(-x, torch.sqrt(sqrt_component)) - torch.atan2(
        l3 * torch.sin(wrist_angle),
        l2 + l3 * torch.cos(wrist_angle))
    return torch.stack([-shoulder_angle, elbow_angle, wrist_angle], dim=-1)
