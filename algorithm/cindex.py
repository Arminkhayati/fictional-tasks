import torch
from tqdm import trange

class Cindex(torch.nn.Module):
    def __init__(self):
        super(Cindex, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, y, y_hat, status):
        if not torch.is_tensor(y):
            y = torch.Tensor(y).to(self.device)
        if not torch.is_tensor(y_hat):
            y_hat = torch.Tensor(y_hat).to(self.device)
        if not torch.is_tensor(status):
            status = torch.Tensor(status).to(self.device)

        N = y.size(0)
        total_pairs = 0
        c = 0

        for i in trange(N):
            a = y[i]
            a_hat = y_hat[i]
            b = y[ i +1:]
            b_hat = y_hat[ i +1:]
            astatus = status[i]
            bstatus = status[ i +1:]

            a_greater = a >= b
            a_hat_greater = a_hat >= b_hat
            bstatus_1 = bstatus == 1

            a_lesser =  a <= b
            a_hat_lesser = a_hat <= b_hat
            astatus_1 = astatus == 1
            c1 = torch.logical_and(a_greater.to("cuda"), a_hat_greater.to("cuda"))
            c1 = torch.logical_and(c1, bstatus_1.to("cuda"))

            c2 = torch.logical_and(a_lesser.to("cuda"), a_hat_lesser.to("cuda"))
            c2 = torch.logical_and(c2, astatus_1.to("cuda"))
            cc = torch.logical_or(c1, c2)
            c += torch.sum(cc.long())

            tp1 = torch.logical_and(a_lesser.to("cuda"), astatus_1.to("cuda"))
            tp2 = torch.logical_and(a_greater.to("cuda"), bstatus_1.to("cuda"))
            tp = torch.logical_or(tp1, tp2)
            total_pairs += torch.sum(tp.long())


        # for i in trange(N):
        #     for j in range(i + 1, N):
        #         a = y[i]
        #         b = y[j]
        #         a_hat = y_hat[i]
        #         b_hat = y_hat[j]
        #         astatus = status[i]
        #         bstatus = status[j]
        #         if (a >= b and a_hat >= b_hat and bstatus == 1) or (a <= b and a_hat <= b_hat and astatus == 1):
        #             c += 1
        #         if (a <= b and astatus==1) or (b <= a and bstatus == 1):
        #             total_pairs += 1

        outcome = c / total_pairs
        return outcome.cpu().item()
