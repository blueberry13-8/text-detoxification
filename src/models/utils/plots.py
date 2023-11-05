import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, val_bleus, val_meteors):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(range(len(train_losses)), train_losses, label='training', marker='o', linestyle='-')
    ax1.plot(range(len(val_losses)), val_losses, label='validation', marker='o', linestyle='-')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoches')
    ax1.legend()

    ax2.plot(range(len(val_bleus)), val_bleus, marker='o', linestyle='-')
    ax2.set_ylabel('BLEU')
    ax2.set_xlabel('Epoches')

    ax3.plot(range(len(val_meteors)), val_meteors, marker='o', linestyle='-')
    ax3.set_ylabel('METEOR')
    ax3.set_xlabel('Epoches')

    plt.show()