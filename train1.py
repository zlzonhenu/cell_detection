from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from detection import *
from utils import CellImageLoad
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from networks import UNet
import argparse

#defefe

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-t",
        "--train_path",
        dest="train_path",
        help="training dataset's path",
        default="/home/hyeonwoo/research/new_research/D4_1/pseudo-labeling/third",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--val_path",
        dest="val_path",
        help="validation data path",
        default="/home/hyeonwoo/research/new_research/D4_1/val",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="save weight path",
        default="/home/hyeonwoo/research/new_research/weight/pseudo/third/best.pth",
    )
    parser.add_argument(
        "-lw",
        "--load_weight_path",
        dest="load_weight_path",
        help="load weight path",
        default="/home/hyeonwoo/research/new_research/weight/pseudo/second/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", default=True, action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=32, type=int
    )
    parser.add_argument(
        "--visdom", dest="vis", help="visdom show", default=True, type=bool
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=800, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )

    args = parser.parse_args()
    return args


class _TrainBase:
    def __init__(self, args):
        ori_paths = self.gather_path(args.train_path, "psuedo-ori")
        gt_paths = self.gather_path(args.train_path, "psuedo-gt")
        data_loader = CellImageLoad(ori_paths, gt_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        self.number_of_traindata = data_loader.__len__()

        ori_paths = self.gather_path(args.val_path, "ori")
        gt_paths = self.gather_path(args.val_path, "gt")
        data_loader = CellImageLoad(ori_paths, gt_paths)
        self.val_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=5, shuffle=False, num_workers=0
        )

        self.save_weight_path = args.weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_weight_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
                args.epochs, args.batch_size, args.learning_rate, args.gpu
            )
        )

        self.net = net

        self.train = None
        self.val = None

        self.N_train = None
        self.optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.criterion = nn.MSELoss()
        self.losses = []
        self.val_losses = []
        self.evals = []
        self.epoch_loss = 0
        self.bad = 0
        self.vis = args.vis

    def gather_path(self, train_paths, mode):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.tif")))
        return ori_paths

    def show_graph(self):
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        plt.plot(x, self.val_losses)
        plt.show()


class TrainNet(_TrainBase):
    def loss_calculate(self, masks_probs_flat, true_masks_flat):
        return self.criterion(masks_probs_flat, true_masks_flat)

    def create_vis_show(self):
        return self.vis.images(
            torch.ones((self.batch_size, 1, 256, 256)), self.batch_size
        )

    def update_vis_show(self, images, window1):
        self.vis.images(images, self.batch_size, win=window1)

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend),
        )

    def update_vis_plot(self, iteration, loss, window1, update_type):
        self.vis.line(
            X=torch.ones((1)).cpu() * iteration,
            Y=torch.Tensor(loss).unsqueeze(0).cpu(),
            win=window1,
            update=update_type,
        )

    def main(self):
        if self.vis:
            import visdom

            HOSTNAME = "localhost"
            PORT = 8097

            self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env="main2")

            vis_title = "ctc"
            vis_legend = ["Loss"]
            vis_epoch_legend = ["Loss", "Val Loss"]

            self.iter_plot = self.create_vis_plot(
                "Iteration", "Loss", vis_title, vis_legend
            )
            self.epoch_plot = self.create_vis_plot(
                "Epoch", "Loss", vis_title, vis_epoch_legend
            )
            self.ori_view = self.create_vis_show()
            self.gt_view = self.create_vis_show()
            self.pred_view = self.create_vis_show()

        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))

            pbar = tqdm(total=self.number_of_traindata)
            self.net.train()
            iteration = 1
            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks = data["gt"]

                if self.gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = self.net(imgs)
                #plt.imshow(imgs[0][0].detach().cpu()), plt.savefig("/home/hyeonwoo/research/new_research/detection/practice/img.png")
                #plt.imshow(true_masks[0][0].detach().cpu()), plt.savefig("/home/hyeonwoo/research/new_research/detection/practice/gt.png")
                #plt.imshow(masks_pred[0][0].detach().cpu()), plt.savefig("/home/hyeonwoo/research/new_research/detection/practice/pred.png")

                masks_probs_flat = masks_pred.view(-1)
                true_masks_flat = true_masks.view(-1)

                loss = self.loss_calculate(masks_probs_flat, true_masks_flat)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iteration += 1
                if self.vis:
                    self.update_vis_plot(
                        iteration, [loss.item()], self.iter_plot, "append"
                    )

                    self.update_vis_show(imgs.cpu(), self.ori_view)
                    self.update_vis_show(masks_pred, self.pred_view)
                    self.update_vis_show(true_masks.cpu(), self.gt_view)

                pbar.update(self.batch_size)
            pbar.close()
            masks_pred = masks_pred.detach().cpu().numpy()
            cv2.imwrite("conf.tif", (masks_pred * 255).astype(np.uint8)[0, 0])
            self.validation(i, epoch)

            if self.bad >= 100:
                print("stop running")
                break
        self.show_graph()

    def validation(self, number_of_train_data, epoch):
        loss = self.epoch_loss / (number_of_train_data + 1)
        print("Epoch finished ! Loss: {}".format(loss))

        self.losses.append(loss)
        if epoch % 10 == 0:
            torch.save(
                self.net.state_dict(),
                str(
                    self.save_weight_path.parent.joinpath(
                        "epoch_weight/{:05d}.pth".format(epoch)
                    )
                ),
            )
        val_loss = eval_net(self.net, self.val_loader, gpu=self.gpu)
        if loss < 0.1:
            print("val_loss: {}".format(val_loss))
            try:
                if min(self.val_losses) > val_loss:
                    print("update best")
                    torch.save(self.net.state_dict(), str(self.save_weight_path))
                    self.bad = 0
                else:
                    self.bad += 1
                    print("bad ++")
            except ValueError:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
            self.val_losses.append(val_loss)
        else:
            print("loss is too large. Continue train")
            self.val_losses.append(val_loss)
        if self.vis:
            self.update_vis_plot(
                iteration=epoch,
                loss=[loss, val_loss],
                window1=self.epoch_plot,
                update_type="append",
            )
        print("bad = {}".format(self.bad))
        self.epoch_loss = 0


if __name__ == "__main__":
    args = parse_args()

    args.train_path = [Path(args.train_path)]
    args.val_path = [Path(args.val_path)]
    # save weight path
    args.weight_path = Path(args.weight_path)

    # define model
    net = UNet(n_channels=1, n_classes=1)
    net.load_state_dict(torch.load(args.load_weight_path, map_location="cuda:0"))
    if args.gpu:
        net.cuda()

    args.net = net

    train = TrainNet(args)

    train.main()