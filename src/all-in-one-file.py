#! git clone https://github.com/mszczesniak02/bachlor_google

#! cp -r /content/bachlor_google/DeepCrack/ .

#! pip install segmentation-models-pytorch

# -------------------- imports -------------------------------

import albumentations as A                              # for augmentation transform

import numpy as np                                      # sci kit specials ;D
import matplotlib.pyplot as plt                         # plots 
from PIL import Image                                   # for opening images as numpy arrays or torch tensors


import torch

from torch.utils.data import Dataset                    # preset class for creating a dataset
from torch.utils.data import random_split               # for splitting datasets into training, test, validation
from torch.utils.data import DataLoader                 # self-explanitory
import segmentation_models_pytorch as smp               # preset model 

from tqdm import tqdm                                   # for the progress bar
import os                                               # for accessing files and setting proper paths to   them 

from torch.utils.tensorboard import SummaryWriter       # tensorboard srv


# ------------ HYPERPARAMETERS ------------------------
DEBUG = True

if DEBUG==True:
   
  MASK_PATH = "../assets/datasets/DeepCrack/train_lab"
  IMAGE_PATH = "../assets/datasets/DeepCrack/train_img"
  DEVICE = "cpu"
  WORKERS = 4

else:
  MASK_PATH = "/content/DeepCrack/train_lab"
  IMAGE_PATH = "/content/DeepCrack/train_img"
  DEVICE="cuda"  
  WORKERS = 2

BATCH_SIZE = 8
PIN_MEMORY = True
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 3

EARLY_STOPPING_PATIENCE = 15

SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
# ----------------------------------------------------

#------------------- DATA SOON-TO-BE PIPELINE -------------------

def fetch_data(path) -> list:
  """Return files as their paths+filename in an array"""

  assert (os.path.exists(path) == True),  "Failure during data fetching"  
      
  result = []
  for file in tqdm(os.listdir(path), desc=f"Loading files from {path} ",unit="File", leave=True):
    fpath = os.path.join(path,file)
    result.append(fpath)
  
  return result


class DeepCrackDataset(Dataset):
  def __init__(self, img_dir, mask_dir, transform=None):
    
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.transform = transform

    # sort values so the file names corespoding to each other are loaded in order
    self.images = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir)] )
    self.masks = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)]) 

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    np_image = np.array(Image.open(self.images[index]))
    np_mask = np.array(Image.open(self.masks[index])) 

   
    if len(np_mask.shape) == 3:
      np_mask = np_mask[:,:,0]

    np_mask = (np_mask > 127).astype(np.uint8)
    
    if self.transform: # if using transforms
      t = self.transform(image=np_image, mask=np_mask)
      np_image = t["image"]
      np_mask = t["mask"]

    # conversion from numpy array convention to tensor via permute, 
    #     then normalizing to [0,1] range, same for mask, only using binary data
    tensor_image = torch.from_numpy(np_image).permute(2, 0, 1).float() / 255.0
    tensor_mask = torch.from_numpy(np_mask).unsqueeze(0).float() 

    return tensor_image,tensor_mask


def get_dataset(img_path = IMAGE_PATH, mask_path = MASK_PATH ):
  
  dataset = DeepCrackDataset(img_path, mask_path, transform=transofrm_train)
  return dataset

def split_dataset(dataset: DeepCrackDataset, train_factor, test_factor, val_factor )->list:
  """Split exising dataset given percentages as [0,1] floats, return list of  """
  train_set_len, test_set_len, val_set_len = int(dataset.__len__() * train_factor), int(dataset.__len__() * test_factor) , int(dataset.__len__() * val_factor)
  train_set, test_set ,val_set = random_split(dataset, [train_set_len, test_set_len, val_set_len])
  
  return [train_set, test_set, val_set]

def show_dataset(data_loader, samples=4):
    counter = 0
    for images, masks in data_loader:        
        fig, axes = plt.subplots(samples, 2, figsize=(8, 12))
        for i in range( samples ):
            
            img = images[i].permute(1, 2, 0).numpy()
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')

            # # Maska
            mask = masks[i, 0].numpy()
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f"Mask {i+1}")
            axes[i, 1].axis('off')
        plt.tight_layout()
        plt.show()
        counter+=1
        
# ---------------------------------------------------------------------

# -------------- AUGMENTATION TRANSFORM ----------------------
transofrm_train = A.Compose([
    A.RandomResizedCrop(size=(256,256),scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),  
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
], seed=np.random.randint(low=1, high=1000))
# ----------------------------------------------------

# ------------------------ LOSS FUNCTIONS --------------------------
class DiceLoss(torch.nn.Module):
  def __init__(self, smooth=1e-6):
    super(DiceLoss,self).__init__()
    self.smooth = smooth
  def forward(self, predictions, targets):
    predictions = torch.sigmoid(predictions)

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()
    dice = (2. * intersection  + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

    return 1-dice
# ----------------------------------------------------

# ----------------------------- METRICS ----------------------------------------------------------------


def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Oblicz metryki segmentacji dla pojedynczego batcha
    
    Args:
        predictions: tensor [B, 1, H, W] - output z modelu (po sigmoid)
        targets: tensor [B, 1, H, W] - ground truth maski
        threshold: próg binaryzacji (default 0.5)
    
    Returns:
        dict z metrykami
    """
    # Binaryzacja
    preds = (predictions > threshold).float()
    targets = targets.float()
    
    # Flatten
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    # True/False Positives/Negatives
    TP = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
    TN = ((preds_flat == 0) & (targets_flat == 0)).sum().float()
    FP = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
    FN = ((preds_flat == 0) & (targets_flat == 1)).sum().float()
    
    # Metryki
    epsilon = 1e-7  # Unikaj dzielenia przez zero
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = TN / (TN + FP + epsilon)
    
    # IoU (Intersection over Union) - NAJWAŻNIEJSZA dla segmentacji!
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = intersection / (union + epsilon)
    
    # Dice Coefficient
    dice = (2 * intersection) / (preds.sum() + targets.sum() + epsilon)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item(),
        'specificity': specificity.item(),
        'iou': iou.item(),
        'dice': dice.item(),
    }


# -------------------------------------------------------------------------------------------------------



# ------------------------------ TRAIN / VALIDATION / TEST / -------------------------


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = .0
    metrics = {
        'iou': [], 'dice': [], 'recall': [], 
        'precision': [], 'f1_score': []
    }
    loop = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (images, masks) in enumerate(loop):
        # move to adequete memory 
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        with torch.no_grad():
            predictions_sigmoid = torch.sigmoid(predictions)
            batch_metrics = calculate_metrics(predictions_sigmoid, masks)
            
            for key in metrics.keys():
                metrics[key].append(batch_metrics[key])
        
        loop.set_postfix({'loss': loss.item()})
    
    loop.close()

    avg_loss = running_loss / len(train_loader)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['loss'] = avg_loss
    

    return avg_metrics

def validate(model, val_loader, criterion,device):
  model.eval()
  running_loss = 0.0
  metrics = {
      'iou': [], 'dice': [], 'recall': [], 
      'precision': [], 'f1_score': [], 'accuracy': []
      }
  with torch.no_grad():
    for images,masks in tqdm(val_loader, desc="Validation", leave=False):
      images = images.to(device)
      masks = masks.to(device)

      predictions = model(images)
      loss = criterion(predictions, masks)

      running_loss += loss.item()

      predictions_sigmoid = torch.sigmoid(predictions)
      batch_metrics = calculate_metrics(predictions_sigmoid, masks)
            
      for key in metrics.keys():
          metrics[key].append(batch_metrics[key])
  
    avg_loss = running_loss / len(val_loader)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics



def make_checkpoint(model, optimizer, best_val_loss, training_loss):
   torch.save({
      'epoch': EPOCHS,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'train_loss': training_loss,
      'val_loss': best_val_loss,} , 'unet_MODEL_save.pth')




def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Kompletna ewaluacja modelu
    
    Returns:
        metrics: dict z metrykami
        predictions: array predykcji [N, H, W]
        ground_truths: array prawdziwych masek [N, H, W]
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Do CPU
            all_predictions.append(predictions.cpu().numpy())
            all_ground_truths.append(masks.cpu().numpy())
    
    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0)[:, 0]  # [N, H, W]
    ground_truths = np.concatenate(all_ground_truths, axis=0)[:, 0]  # [N, H, W]
    
    # Binaryzacja
    pred_binary = (predictions > threshold).astype(np.float32)
    gt_binary = (ground_truths > 0.5).astype(np.float32)
    
    # ========================================
    # OBLICZ METRYKI
    # ========================================
    ious = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for pred, gt in zip(pred_binary, gt_binary):
        # Confusion matrix
        tp = (pred * gt).sum()
        fp = (pred * (1 - gt)).sum()
        fn = ((1 - pred) * gt).sum()
        tn = ((1 - pred) * (1 - gt)).sum()
        
        # IoU (Intersection over Union)
        iou = tp / (tp + fp + fn + 1e-6)
        ious.append(iou)
        
        # Precision (jakość detekcji)
        precision = tp / (tp + fp + 1e-6)
        precisions.append(precision)
        
        # Recall (czułość)
        recall = tp / (tp + fn + 1e-6)
        recalls.append(recall)
        
        # F1 Score (harmonic mean)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        f1_scores.append(f1)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        accuracies.append(accuracy)
    
    metrics = {
        'iou_mean': np.mean(ious),
        'iou_std': np.std(ious),
        'iou_median': np.median(ious),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'accuracy_mean': np.mean(accuracies),
        'ious': ious,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
    }
    
    return metrics, predictions, ground_truths



#--------------------------------------------------------------------------------

# ----------------------- MAIN FUNCTION -----------------------------------------



def main()-> int:

    dataset = get_dataset(IMAGE_PATH, MASK_PATH)
    [train_set, test_set, val_set] = split_dataset(dataset, .8, .1, .1)

    train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader( val_set, batch_size=BATCH_SIZE , shuffle=False, num_workers=WORKERS    , pin_memory=PIN_MEMORY)

    print("Loading datasets...")    
    print(f"   Train: {len(train_set)} images")
    print(f"   Val:   {len(val_set)} images")
    print(f"   Test:  {len(test_set)} images")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    model = model.to(device)       


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB")



    criterion = DiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maksymalizuj IoU
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=True,
        min_lr=1e-7
    )

    print(f"\nTraining configuration:")
    print(f"   Optimizer: Adam")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Weight decay: {WEIGHT_DECAY}")
    print(f"   Scheduler: ReduceLROnPlateau (patience={SCHEDULER_PATIENCE})")
    print(f"   Early stopping: patience={EARLY_STOPPING_PATIENCE}")
    print(f"   Epochs: {EPOCHS}")




    epochs = EPOCHS
    best_val_iou = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 80}")
        
        # TRAIN
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # VALIDATE
        val_metrics = validate(model, val_loader, criterion, device)
        
        # SCHEDULER
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['iou'])

        

       # ========================================
        # PRINT METRICS
        # ========================================
        print(f"\nTraining:")
        print(f"   Loss: {train_metrics['loss']:.4f}")
        print(f"   IoU:  {train_metrics['iou']:.4f}")
        print(f"   Dice: {train_metrics['dice']:.4f}")
        
        print(f"\nValidation:")
        print(f"   Loss:      {val_metrics['loss']:.4f}")
        print(f"   IoU:       {val_metrics['iou']:.4f} {'✅ NEW BEST!' if val_metrics['iou'] > best_val_iou else ''}")
        print(f"   Dice:      {val_metrics['dice']:.4f}")
        print(f"   Recall:    {val_metrics['recall']:.4f}")
        print(f"   Precision: {val_metrics['precision']:.4f}")
        print(f"   F1-Score:  {val_metrics['f1_score']:.4f}")
        
        print(f"\n LR: {current_lr:.6f}")
        
        # ========================================
        # SAVE BEST MODEL
        # ========================================
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }
            
            torch.save(checkpoint, f'best_model_iou_{best_val_iou:.4f}.pth')
            print(f"\n Model saved: best_model_iou_{best_val_iou:.4f}.pth")
        else:
            patience_counter += 1
            print(f"\n No improvement for {patience_counter} epoch(s)")
        
        # ========================================
        # EARLY STOPPING
        # ========================================
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered!")
            print(f"   No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            print(f"   Best IoU: {best_val_iou:.4f} at epoch {best_epoch + 1}")
            break
    
    # ========================================
    # FINAL EVALUATION ON TEST SET
    # ========================================
    print(f"\n{'=' * 80}")
    print("Final evaluation on test set...")
    print(f"{'=' * 80}")
    
    # Load best model
    checkpoint = torch.load(f'best_model_iou_{best_val_iou:.4f}.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"   IoU:       {test_metrics['iou']:.4f}")
    print(f"   Dice:      {test_metrics['dice']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    
    # Save to TensorBoard
    # writer.add_text('Final_Test_Metrics', str(test_metrics))
    
    # writer.close()
    
    print(f"\n{'=' * 80}")
    print(" TRAINING COMPLETED!")
    print(f"   Best Validation IoU: {best_val_iou:.4f} (epoch {best_epoch + 1})")
    print(f"   Test IoU: {test_metrics['iou']:.4f}")
    print(f"   Model saved as: best_model_iou_{best_val_iou:.4f}.pth")
    print(f"{'=' * 80}\n")
    
    return 0

# -----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  main()
else:
  pass

