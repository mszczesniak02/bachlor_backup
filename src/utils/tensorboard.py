import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(writer, model, val_loader, device, epoch, num_samples=4):
    """
    Wizualizuje predykcje modelu w TensorBoard.
    Pokazuje: obraz wejściowy, maskę GT, predykcję modelu.
    
    Args:
        writer: SummaryWriter do TensorBoard
        model: model segmentacyjny
        val_loader: DataLoader walidacyjny
        device: urządzenie (cuda/cpu)
        epoch: numer epoki
        num_samples: liczba przykładów do wizualizacji
    """
    model.eval()
    
    # Pobierz jeden batch z val_loader
    images, masks = next(iter(val_loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].to(device)
    
    with torch.no_grad():
        predictions = model(images)
        predictions_sigmoid = torch.sigmoid(predictions)
        predictions_binary = (predictions_sigmoid > 0.5).float()
    
    # Przenieś na CPU i konwertuj do numpy
    images_np = images.cpu()
    masks_np = masks.cpu()
    predictions_np = predictions_binary.cpu()
    
    # Dla każdego sample osobno
    for i in range(num_samples):
        # Obraz wejściowy (denormalizacja jeśli trzeba)
        img = images_np[i]
        
        # Maska GT
        mask_gt = masks_np[i].repeat(3, 1, 1)  # Powtórz dla 3 kanałów RGB
        
        # Predykcja
        pred = predictions_np[i].repeat(3, 1, 1)
        
        # Złóż obok siebie: [obraz | maska GT | predykcja]
        comparison = torch.cat([img, mask_gt, pred], dim=2)  # Konkatenuj wzdłuż szerokości
        
        # Dodaj do TensorBoard
        writer.add_image(f'Predictions/sample_{i}', comparison, epoch)
    
    # Alternatywnie: grid wszystkich przykładów
    grid_images = vutils.make_grid(images_np, nrow=num_samples, normalize=True)
    grid_masks = vutils.make_grid(masks_np.repeat(1, 3, 1, 1), nrow=num_samples, normalize=True)
    grid_preds = vutils.make_grid(predictions_np.repeat(1, 3, 1, 1), nrow=num_samples, normalize=True)
    
    # Stack wertykalnie
    grid_combined = torch.cat([grid_images, grid_masks, grid_preds], dim=1)
    writer.add_image('Predictions/grid_all', grid_combined, epoch)
    
    model.train()
