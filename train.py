import json
import os

import torch

from dataset import SketchDataset
from model import SketchModel
from parameters import HParams


##################### hyperparameters #####################


####################################################################################
####################################################################################

if __name__ == "__main__":
    hp = HParams()
    dataloader = SketchDataset(hp)
    print(hp.foldername)

    model = SketchModel(hp)

    for step in range(100001):
        batch_data, batch_len = dataloader.train_batch()
        kl_cost, recons_loss, loss, curr_learning_rate, curr_kl_weight = model.train_model(batch_data, batch_len, step)

        print('Step:{} ** Current_LR:{} ** Current_KL:{} ** KL_Loss:{} '
              '** Recons_Loss:{} ** Total_loss:{}'.format(step, curr_learning_rate, curr_kl_weight,
                                                          kl_cost, recons_loss, loss))
        if (step + 1) % 100 == 0:
            model.generation(dataloader, step, number_of_sample=20, condition=False)

        if (step + 1) % 1000 == 0:
            if not os.path.exists(hp.model_folder):
                os.makedirs(hp.model_folder)
            json.dump(vars(hp), open(os.path.join(hp.model_folder, 'config.json'), 'w'), indent=4)
            torch.save(model.state_dict(), os.path.join(hp.model_folder, f'sketch_{hp.codename}_{step}.pth'))
