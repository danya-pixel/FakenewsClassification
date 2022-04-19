import matplotlib.pyplot as plt
from IPython.display import clear_output
from wordcloud import WordCloud

def show_progress(epoch: int, train_history: list, val_history: list) -> None:
    """Show learning progress at two plots

    Args:
        epoch (int): current epoch
        train_loss (list): list of train loss history
        train_acc (list): list of train accuracy history
        val_acc (list): list of val accuracy history
    """
    clear_output(wait=True)
    epochs_list = list(range(epoch+1))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, constrained_layout=True, figsize=(20, 5))
    
    val_acc = val_history['acc']
    fig.suptitle(
        f'Epoch {epoch}, max val acc {(max(val_acc) if val_acc else 0):.3f}', fontsize=16)
    ax1.set_title('loss')
    ax1.set_xlabel('time (epochs)')
    ax1.set_ylabel('loss')
    ax1.plot(epochs_list, train_history['loss'], c='darkblue', marker='o', lw=5)
    ax1.plot(epochs_list, val_history['loss'], c='green', marker='o', lw=5)
    ax1.legend(['train', 'valid'])
    ax2.set_title('accuracy')
    ax2.set_xlabel('time (epochs)')
    ax2.plot(epochs_list, train_history['acc'], c='darkblue', marker='o', lw=5)
    ax2.plot(epochs_list, val_history['acc'], c='green', marker='o', lw=5)
    ax2.legend(['train', 'valid'])
    plt.show()


def get_wordcloud(data, title):
    """Draws a wordcloud of words from all sentences

    Args:
        data (iterable): List of sentences 
        title (str): Title for wordcloud
    """
    all_words = " ".join(lemma for lemma in data)

    wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(all_words)
    fig = plt.figure(figsize=(30,10), facecolor='white')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title, fontsize=30)
    plt.tight_layout(pad=0)
    plt.show()