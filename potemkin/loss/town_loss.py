import torch

class DistanceLoss(torch.nn.Module):


    def forward(self, input, target):
        """
        Compute a distance-based loss for towns.
        For false-positives: The point resides within a frame but not in a window. The loss is therefore
        :param input:
        :param target:
        :return:
        """

        frame_containment = input["crisp_frame_containment"]
        inner_frame_distances_when_inside = input["inner_frame_distance"]*frame_containment
        outer_frame_distances_when_outside = input["outer_frame_distance"]*(1-frame_containment)
        house_containment = input["crisp_house_containment"]
        containment = input["crisp_containment"].unsqueeze(-1)
        inner_window_distances_when_inside = torch.sum(input["inner_window_distances"] * input["crisp_window_containment"],
                                           dim=-1) * frame_containment
        outer_window_distances_when_outside = torch.sum(input["outer_window_distances"] * (1 - input["crisp_window_containment"]),
                                           dim=-1) * frame_containment
        belongs = target.unsqueeze(-1)
        fn_loss = torch.min(belongs * (1 - containment) * (outer_frame_distances_when_outside + inner_window_distances_when_inside), dim=-1)[0]
        fp_loss = torch.max((~belongs) * containment * (inner_frame_distances_when_inside + outer_window_distances_when_outside), dim=-1)[0]

        return torch.mean(fp_loss+fn_loss)