import torch

class DistanceLoss(torch.nn.Module):


    def forward(self, input, target):
        d = input["output"]
        frame_containment = d["frame_containment"]
        inner_frame_distances = d["inner_frame_distance"]
        outer_frame_distances = d["outer_frame_distance"]
        house_containment = d["house_containment"]
        containment = d["containment"].unsqueeze(-1)
        inner_window_distances = torch.sum(d["inner_window_distances"] * d["window_containment"],
                                           dim=-1) * frame_containment
        outer_window_distances = torch.sum(d["outer_window_distances"] * (1 - d["window_containment"]),
                                           dim=-1) * frame_containment
        belongs = target.unsqueeze(-1)
        fn_loss = torch.min(belongs * (1 - containment) * (outer_frame_distances + inner_window_distances), dim=-1)[0]
        fp_loss = torch.max((~belongs) * containment * (inner_frame_distances + outer_window_distances), dim=-1)[0]

        return torch.mean(fp_loss+fn_loss)