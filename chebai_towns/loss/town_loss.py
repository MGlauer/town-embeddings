import torch

class DistanceLoss(torch.nn.Module):


    def forward(self, input, target):

        loss = 0
        d = input["output"]
        for town_distance, belongs, is_contained in zip(d["distances"], target.T, d["containment"]):
            belongs_loss = []
            neg_belongs_loss = []
            for house_distance in town_distance:
                inner_window_dis = 0
                closest_outer_window = torch.tensor(1)
                if house_distance["inner_window_distances"] is not None:
                    inner_window_dis = torch.sum(house_distance["inner_window_distances"], dim=0)
                    closest_outer_window = torch.sum(house_distance["outer_window_distances"], dim=0)
                belongs_loss.append(
                    belongs * (1 - is_contained) * (house_distance["outer_frame_distance"] + inner_window_dis))
                neg_belongs_loss.append(
                    (~belongs) * is_contained * (house_distance["inner_frame_distance"] + closest_outer_window))
            loss += torch.min(torch.stack(belongs_loss, dim=0), dim=0)[0]
            loss += torch.max(torch.stack(neg_belongs_loss, dim=0), dim=0)[0]

        return torch.mean(loss)