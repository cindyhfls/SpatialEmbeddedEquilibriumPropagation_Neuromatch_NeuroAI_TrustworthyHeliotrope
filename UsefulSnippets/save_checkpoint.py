checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': w_optimizer.state_dict(), 
        'test_accuracy': test_acc,
        'cfg':cfg,
    }
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
