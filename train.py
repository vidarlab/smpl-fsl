import argparse
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils

from dataset import EpisodeDataset
from model import PartMatchingTransformer

# Configure logging
def setup_logger(run_id):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    log_dir = f"logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{log_dir}/{run_id}.log")
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_model(model, dataloaders, optimizer, scheduler, num_epochs, cls_criterion, logger):
    
    scaler = torch.cuda.amp.GradScaler()
    since = time.time()

    # Create a temporary directory to save training checkpoints
    base = f"checkpoints/{run_id}"
    os.makedirs(base, exist_ok=True)
    latest_model_params_path = os.path.join(base, 'latest.pt')
    max_accuracy = 0.0

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        logger.info('-' * 10)

        for split in ['train', 'val']:
            if split == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            dataloader = dataloaders[split]
            num_examples = 0
            running_cls_loss = 0.0
            running_corrects = 0
            
            with torch.set_grad_enabled(split=='train'):
            # Iterate over data.
                for i, batch in enumerate(dataloader):

                    targets = batch['targets'][0].to(device)
                    features = batch['features'][0].to(device)
                    source_ids = batch['source_ids'][0].to(device)
                    part_indices = batch['part_indices'][0].to(device)
                    padding = batch['is_padding'][0].to(device)
                    ways = torch.max(targets) + 1

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        
                        logits = model(features, source_ids, padding)
                        preds = torch.argmax(logits, dim=1)
                        cls_loss = cls_criterion(logits, targets) / torch.log(ways)

                        if split == 'train':
                            if i % 10 == 0:  # Log less frequently to avoid too many logs
                                logger.debug(f"Train batch {i}, loss: {cls_loss.item():.4f}")
                            scaler.scale(cls_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            if scheduler is not None:
                                try:
                                    scheduler.step()
                                except:
                                    logger.error("Error in scheduler.step()")
                                    pass

                    # statistics
                    running_cls_loss += cls_loss.item() * features.size(0)
                    running_corrects += torch.sum(preds == targets)
                    num_examples += targets.size(0)

            epoch_cls_loss = running_cls_loss / num_examples
            epoch_acc = running_corrects.double() / num_examples
            logger.info(f"{split} - Epoch: {epoch}, Loss: {epoch_cls_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        torch.save(model.state_dict(), latest_model_params_path)
        if split == 'val' and epoch_acc > max_accuracy:
            max_accuracy = epoch_acc
            logger.info(f"New best model with accuracy: {epoch_acc:.4f}")
            torch.save(model.state_dict(), os.path.join(base, 'best.pt'))

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best validation accuracy: {max_accuracy:.4f}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Classification Hotels Training", allow_abbrev=False
        )
    parser.add_argument("--data_dir", default='data/', type=str, help="location to features")
    parser.add_argument("--dataset", default='hotels', type=str, help="dataset")
    parser.add_argument("--part_encoder", default='dinov2', type=str, help="part encoding method")
    parser.add_argument("--num_layers", default=4, type=int, help="number of transformer layers")
    parser.add_argument("--num_heads", default=8, type=int, help="number of attention heads")
    parser.add_argument("--input_feat_dim", default=768, type=int, help="dimension of precomputed part embeddings")
    parser.add_argument("--dim_model", default=768, type=int, help="dimension of model")
    parser.add_argument("--dim_ff", default=768 * 2, type=int, help="dimension of feedforward layer")
    parser.add_argument("--temperature", default=20., type=float, help="temperature parameter")
    parser.add_argument("--pos_encodings", default=True, type=str2bool, help="pos encodings")
    parser.add_argument("--dropout", default=.1, type=float, help="dropout rate")
    parser.add_argument("--token_dropout", default=.2, type=float, help="dropout rate")
    parser.add_argument("--noise_std", default=.1, type=float, help="dropout rate")
    parser.add_argument("--lr", default=5e-3, type=float, help="learning rate")
    parser.add_argument("--smoothing", default=.1, type=float, help="smoothing value")
    parser.add_argument("--momentum", default=.9, type=float, help="momentum")
    parser.add_argument("--num_epochs", default=100, type=int, help="number of epochs for training")
    parser.add_argument("--num_workers", default=12, type=int, help="num dataloading workers")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    args = parser.parse_args()

    # Generate unique run ID based on timestamp
    run_id = f"run_{int(time.time())}"
    
    # Setup logger
    logger = setup_logger(run_id)

    # Log all parameters
    logger.info("Starting training with parameters:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    PARAMS = vars(args)

    SHOT_RANGE = [1, 5]
    WAY_RANGE = [5, 20]

    seed = PARAMS['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    PARAMS['dataset'] = PARAMS['dataset'].lower()

    if 'hotel' in PARAMS['dataset']:
        train_dataset = EpisodeDataset(data_dir=PARAMS['data_dir'], dataset=PARAMS['dataset'], part_encoder=PARAMS['part_encoder'], split='train', shots=SHOT_RANGE, ways=WAY_RANGE, 
                                       token_dropout=PARAMS['token_dropout'])
        logger.info(f'Number of images in train split: {len(train_dataset)}')
        val_dataset = EpisodeDataset(data_dir=PARAMS['data_dir'], dataset=PARAMS['dataset'], part_encoder=PARAMS['part_encoder'], split='val', shots=SHOT_RANGE[-1], ways=WAY_RANGE[-1])
        logger.info(f'Number of images in val split: {len(val_dataset)}')    
    else:
        raise ValueError("Dataset not supported")
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=int(len(train_dataset) // (SHOT_RANGE[0] + SHOT_RANGE[1]) / 2))
    val_sampler = torch.utils.data.RandomSampler(val_dataset, num_samples=int(len(val_dataset) // (SHOT_RANGE[0] + SHOT_RANGE[1]) / 2))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=PARAMS['num_workers'], sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=PARAMS['num_workers'], sampler=val_sampler)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    model = PartMatchingTransformer(num_layers=PARAMS['num_layers'], num_heads=PARAMS['num_heads'], dim_model=PARAMS['dim_model'], dim_ff=PARAMS['dim_ff'], dropout=PARAMS['dropout'],
                                      num_sources=WAY_RANGE[-1] + 1, dim_token=PARAMS['input_feat_dim'], init_scalar=PARAMS['temperature'], pos_encodings=PARAMS['pos_encodings'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=PARAMS["smoothing"], reduction='mean')
    optimizer_ft = optim.SGD(model.parameters(), lr=PARAMS["lr"])
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer_ft, max_lr=PARAMS["lr"], epochs=PARAMS["num_epochs"], div_factor=10, steps_per_epoch=len(train_dataset), final_div_factor=10000, pct_start=.1, anneal_strategy='cos')
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=PARAMS["num_epochs"], cls_criterion=criterion, logger=logger)