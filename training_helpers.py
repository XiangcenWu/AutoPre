import torch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric


th = AsDiscrete(threshold=0.5)
def post_process(output):
    output = torch.sigmoid(output)
    return th(output)
dm = DiceMetric(reduction='mean')
def dice_metric(y_pred, y_true, post=False, mean=False):
    """Calculate the dice score (accuracy) give the prediction from a trained segmentation network
    

    Args:
        y_pred (torch.tensor): output from the segmentation network 
        y_true (torch.tensor): ground truth of the output
        post (bool, optional): whether to do the post process (threshold, simoid etc...) 
                               before calculate the dice. Defaults to False.
        mean (bool, optional): calculate the mean of a batch of dice score. Defaults to False.

    Returns:
        torch.tensor: dice score of a single number or a batch of numbers
    """
    if post:
        y_pred = post_process(y_pred)


    return dm(y_pred, y_true).mean() if mean else dm(y_pred, y_true)


def train_seg_net_h5(
        seg_model, 
        seg_loader,
        seg_optimizer,
        seg_loss_function,
        device='cpu',
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    seg_model.train()
    
    step = 0.
    loss_a = 0.
    for batch in seg_loader:

        img, label = batch["image"].to(device), batch["label"].to(device)
        # print(img.shape, label.shape)
        # forward pass and calculate the selection


        # forward pass of selected data
        output = seg_model(img)
        loss = seg_loss_function(output, label)
        # print(loss.item())
        

        loss.backward()
        seg_optimizer.step()
        seg_optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch


def test_seg_net_h5(seg_model, test_loader, device):


    seg_model.eval()
    performance_a = 0.
    step = 0.

    for batch in test_loader:

        img, label = batch["image"].to(device), batch["label"].to(device)


        with torch.no_grad():
            seg_output = seg_model(img)
            # print(seg_output.shape, seg_output[0, 0, 100, 100, :10])
            # seg_output = post_process(seg_output)
            # print(seg_output.shape, seg_output[0, 0, 100, 100, :10])
            performance = dice_metric(seg_output, label, post=True, mean=True)
            # print(performance)


            performance_a += performance.item()
            step += 1.

    performance_of_this_epoch = performance_a / step

    return performance_of_this_epoch
