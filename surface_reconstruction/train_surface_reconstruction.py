# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon_dataset as dataset
import numpy as np
import models.Net as Net
from models.losses import Loss
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import utils.utils as utils
import surface_recon_args
# get training parameters
args = surface_recon_args.get_train_args()
print(args.file_name)

file_path = os.path.join(args.dataset_path, args.file_name)
logdir = os.path.join(args.logdir, args.file_name.split('.')[0])
# set up logging
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)

# get data loaders
torch.manual_seed(0)  #change random seed for training set (so it will be different from test set
np.random.seed(0)
train_set = dataset.ReconDataset(file_path, args.n_points, args.n_iterations)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0,
                                               pin_memory=True)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# get model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:" + str(args.gpu_idx))

Net = Net.Network(in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim, nl=args.nl, decoder_n_hidden_layers=args.decoder_n_hidden_layers,
                  init_type=args.init_type, neuron_type=args.neuron_type, sphere_init_params=args.sphere_init_params)
if args.parallel:
    if (device.type == 'cuda'):
        Net = torch.nn.DataParallel(Net)

n_parameters = utils.count_parameters(Net)
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

# Setup Adam optimizers
optimizer = optim.Adam(Net.parameters(), lr=args.lr, betas=(0.9, 0.999))
num_batches = len(train_dataloader)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[num_batches*0.45,num_batches*0.6,num_batches*0.7,num_batches*0.8,num_batches*0.9], gamma=0.18)#LR=2e-4
Net.to(device)
criterion = Loss(weights=args.loss_weights)

# For each batch in the dataloader
start = time.time()
for batch_idx, data in enumerate(train_dataloader):
    Net.zero_grad()
    Net.train()

    mnfld_points,near_points = \
        data['points'].to(device), data['near_points'].to(device)

    mnfld_points.requires_grad_()
    near_points.requires_grad_()

    output_pred = Net(None, mnfld_points,near_points)

    loss_dict = criterion(output_pred, mnfld_points, None,near_points)
    lr = torch.tensor(optimizer.param_groups[0]['lr'])
    loss_dict["lr"] = lr
    utils.log_losses(log_writer_train, batch_idx, num_batches, loss_dict)

    loss_dict["loss"].backward()
    optimizer.step()

    #Output training stats
    if batch_idx % 100 == 0:
        weights = criterion.weights
        utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
        utils.log_string('Iteration: {:4d}/{} ({:.0f}%) Loss: {:.3e} = L_Dirichlet: {:.3e} + '
                'L_NonMnfld: {:.3e} + L_Neumann: {:.3e} + L_MA: {:.3e}'.format(
             batch_idx, len(train_set), 100. * batch_idx / len(train_dataloader),
                    loss_dict["loss"].item(), weights[0]*loss_dict["dirichlet_term"].item(), weights[1]*loss_dict["inter_term"].item(),
                    weights[2]*loss_dict["neumann_term"].item(),weights[3]*loss_dict["ma_term"].item()), log_file)
        utils.log_string('Iteration: {:4d}/{} ({:.0f}%) Unweighted L_s : L_Dirichlet: {:.3e},  '
                'L_NonMnfld: {:.3e},  L_Neumann: {:.3e},  L_MA: {:.3e}'.format(
             batch_idx, len(train_set), 100. * batch_idx / len(train_dataloader),
                    loss_dict["dirichlet_term"].item(), loss_dict["inter_term"].item(),
                    loss_dict["neumann_term"].item(),loss_dict["ma_term"].item()), log_file)
        utils.log_string('', log_file)
    scheduler.step()

    # save model
    if batch_idx == num_batches - 1:
        utils.log_string("saving model to file :{}".format('model_%d.pth' % (num_batches)),
                         log_file)
        torch.save(Net.state_dict(),
                   os.path.join(model_outdir,'model_%d.pth' % (num_batches)))
end = time.time()
print('time:',(end - start)/60)
